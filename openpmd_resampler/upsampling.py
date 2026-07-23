"""
Kernel-density upsampling of macroparticles by antithetic splitting.

The mergers in this package go many -> few; this module goes one -> many, for
the opposite problem: a dataset with too few macroparticles for good
post-processing statistics. Each parent macroparticle of weight w is split into
n daughters of weight w/n, scattered around the parent by Gaussian kernel noise
whose per-coordinate width is the local weighted standard deviation of the
phase space (estimated per spatial cell) times a user bandwidth, so the
daughter cloud reproduces the parent's local phase-space density.

Physics constraint. For massive particles, splitting one macroparticle into
distinct daughters while conserving weight, momentum AND energy exactly is
impossible: p(E) = sqrt(E^2 - m^2) is concave in E, so at fixed total energy
any nontrivial spread of the daughters strictly decreases the total momentum
magnitude (equivalently E(p) is convex, so spreading momentum at fixed mean
raises the total energy). This is why the Vranic 2015 construction only ever
maps many -> 2 and never 1 -> many. The contract here is therefore:

    * total weight is conserved EXACTLY (per parent, by construction);
    * total momentum is conserved EXACTLY (per parent, by construction) -- the
      daughters are drawn in antithetic pairs (+delta, -delta) so their
      per-parent displacement sums to zero, up to float32 rounding;
    * total energy is conserved only to second order in the kernel bandwidth.
      The error is always positive (convexity of E(p)), shrinks as the
      bandwidth -> 0, and is logged.

The numerics mirror vranic.py: torch float32 on the chosen device (default the
GPU when present, both NVIDIA CUDA and AMD ROCm), a fixed generator seed for
reproducibility (CPU and GPU streams differ, matching the repo's
non-bit-reproducibility caveats), and float64 conservation errors accumulated
on the CPU for logging. The output is n times the input, so the daughters are
generated in chunks of parents sized to the free GPU memory; the per-cell
statistics are computed once on the full dataset before chunking.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .log import logger
from .vranic import _resolve_device, _uniform_bin_indices

try:
    import torch
except ImportError:  # torch is only needed for kernel upsampling
    torch = None

POSITION_COLUMNS = ["position_x_m", "position_y_m", "position_z_m"]
MOMENTUM_COLUMNS = ["momentum_x_mev_c", "momentum_y_mev_c", "momentum_z_mev_c"]
PHASE_COLUMNS = POSITION_COLUMNS + MOMENTUM_COLUMNS

# Rough transient GPU bytes per daughter at the peak of a chunk, used to size
# the chunks of parents (each parent expands into upsampling_factor daughters).
BYTES_PER_DAUGHTER = 96

SEED = 77125


class KernelUpsampler:
    def __init__(self, df: pd.DataFrame, mass_mev_c2: float, weight_column: str = "weights"):
        self.df = df
        self.mass = mass_mev_c2  # 0.0 for photons
        self.weight_column = weight_column

    def upsample(
        self,
        upsampling_factor: int,
        spatial_bins: Tuple[int, int, int] = (16, 16, 16),
        position_bandwidth: float = 0.1,
        momentum_bandwidth: float = 0.1,
        device: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Split each parent macroparticle into upsampling_factor daughters.

        upsampling_factor: number of daughters per parent, n >= 2; each carries
            weight w/n, so the output has exactly n times the input rows.
        spatial_bins: number of position bins along (x, y, z) used to estimate
            the local weighted phase-space spread.
        position_bandwidth, momentum_bandwidth: kernel widths as a fraction of
            the local weighted standard deviation, for the three position and
            the three momentum coordinates respectively. 0.0 disables the
            perturbation of those coordinates (daughters share the parent value).
        device: PyTorch device string, e.g. "cuda", "cuda:1" or "cpu". The
            default picks the GPU when one is available (NVIDIA and AMD ROCm
            builds of PyTorch both expose it as "cuda") and the CPU otherwise.
            The n-fold larger output is generated in chunks of parents sized to
            the free GPU memory, which does not change the result.
        """
        if torch is None:
            raise ImportError(
                "Kernel upsampling requires PyTorch; install the 'pytorch-gpu' (or"
                " 'pytorch' for CPU-only) package in the pixi environment."
            )
        upsampling_factor = int(upsampling_factor)
        if upsampling_factor < 2:
            raise ValueError("upsampling_factor must be at least 2.")
        if position_bandwidth < 0.0 or momentum_bandwidth < 0.0:
            raise ValueError("bandwidths must be non-negative.")

        torch_device = _resolve_device(device)
        logger.info("Kernel upsampling on device '%s'.\n", torch_device)

        phase_np = self.df[PHASE_COLUMNS].to_numpy(np.float32)
        weight_np = self.df[self.weight_column].to_numpy(np.float32)
        number_initial = phase_np.shape[0]

        # Conservation totals for the log, in float64 on the CPU.
        weight64 = weight_np.astype(np.float64)
        momentum64 = phase_np[:, 3:].astype(np.float64)
        total_weight_before = weight64.sum()
        total_momentum_before = weight64 @ momentum64
        total_energy_before = weight64 @ np.sqrt(
            np.einsum("ij,ij->i", momentum64, momentum64) + self.mass**2
        )
        del momentum64

        # Per spatial cell weighted standard deviation of every phase-space
        # coordinate, computed once on the full dataset, in two passes to
        # avoid catastrophic cancellation: E[x^2] - E[x]^2 loses precision in
        # float32 whenever a coordinate's mean dominates its spread (e.g. a
        # position offset far from the origin), silently rounding the
        # variance to zero. Instead, the first pass gets the weighted mean
        # per cell, and the second pass accumulates Sum(w*(x-mean)^2), whose
        # terms are already O(spread^2) and don't cancel.
        number_of_cells = int(np.prod(spatial_bins))
        phase = torch.tensor(phase_np, device=torch_device)
        weight = torch.tensor(weight_np, device=torch_device)
        cell_key = torch.zeros(number_initial, dtype=torch.long, device=torch_device)
        for axis in range(3):
            cell_key.mul_(spatial_bins[axis]).add_(
                _uniform_bin_indices(phase[:, axis], spatial_bins[axis])
            )

        zeros = lambda *shape: torch.zeros(*shape, dtype=phase.dtype, device=torch_device)
        cell_weight = zeros(number_of_cells).index_add_(0, cell_key, weight)
        cell_sum = zeros((number_of_cells, 6)).index_add_(0, cell_key, weight[:, None] * phase)
        tiny = torch.finfo(phase.dtype).tiny
        inverse_weight = 1.0 / cell_weight.clamp_min(tiny)
        mean = cell_sum * inverse_weight[:, None]
        centered = phase - mean[cell_key]
        cell_sum_sq = zeros((number_of_cells, 6)).index_add_(
            0, cell_key, weight[:, None] * centered**2
        )
        variance = (cell_sum_sq * inverse_weight[:, None]).clamp_min(0.0)
        cell_sigma = torch.sqrt(variance)  # (number_of_cells, 6)
        del cell_weight, cell_sum, cell_sum_sq, mean, centered, variance, phase, weight

        # Kernel width per coordinate: the position bandwidth for the three
        # position coordinates, the momentum bandwidth for the three momentum
        # ones. A zero bandwidth leaves those coordinates unperturbed.
        bandwidth = torch.tensor(
            [position_bandwidth] * 3 + [momentum_bandwidth] * 3,
            dtype=torch.float32,
            device=torch_device,
        )

        generator = torch.Generator(device=torch_device).manual_seed(SEED)

        # The output is n times the input, so generate the daughters in chunks
        # of parents sized to the free GPU memory (one chunk on the CPU). The
        # per-cell statistics above are shared by every chunk.
        chunk_target = number_initial
        if torch_device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(torch_device)
            per_parent = max(upsampling_factor * BYTES_PER_DAUGHTER, 1)
            chunk_target = max(int(free_bytes * 0.45) // per_parent, 1)
        chunk_bounds = np.append(np.arange(0, number_initial, chunk_target), number_initial)
        if chunk_bounds.size > 2:
            logger.info(
                "Processing %d chunks of parents to fit the GPU memory.\n",
                chunk_bounds.size - 1,
            )

        position_list, momentum_list, weight_list = [], [], []
        for first, last in zip(chunk_bounds[:-1], chunk_bounds[1:]):
            parents = torch.tensor(phase_np[first:last], device=torch_device)
            parent_weight = torch.tensor(weight_np[first:last], device=torch_device)
            sigma = cell_sigma[cell_key[first:last]]  # (chunk, 6)
            daughters = self._split_parents(
                parents, sigma, bandwidth, upsampling_factor, generator, torch_device
            )
            daughter_weight = (parent_weight / upsampling_factor).repeat_interleave(
                upsampling_factor
            )
            position_list.append(daughters[:, :3].cpu().numpy())
            momentum_list.append(daughters[:, 3:].cpu().numpy())
            weight_list.append(daughter_weight.cpu().numpy())

        upsampled_position = np.concatenate(position_list)
        upsampled_momentum = np.concatenate(momentum_list)
        upsampled_weight = np.concatenate(weight_list)

        self.log_upsample_statistics(
            number_initial,
            upsampling_factor,
            upsampled_weight,
            upsampled_momentum,
            total_weight_before,
            total_momentum_before,
            total_energy_before,
        )

        return self.build_dataframe(upsampled_position, upsampled_momentum, upsampled_weight)

    def _split_parents(
        self,
        parents: "torch.Tensor",
        sigma: "torch.Tensor",
        bandwidth: "torch.Tensor",
        upsampling_factor: int,
        generator: "torch.Generator",
        torch_device: "torch.device",
    ) -> "torch.Tensor":
        """
        Split one chunk of parents into upsampling_factor daughters each.

        parents and sigma are (chunk, 6); the daughters are returned flattened
        to (chunk * upsampling_factor, 6) with parent i occupying the rows
        [i * n : (i + 1) * n]. The Gaussian displacements are drawn in
        antithetic pairs (+delta, -delta), with a single zero-displacement
        daughter when n is odd, so the per-parent sum of displacements vanishes
        exactly and total position and momentum are conserved per parent.
        """
        number_parents = parents.shape[0]
        half = upsampling_factor // 2

        # Written into a preallocated tensor rather than torch.cat of parts,
        # so the peak holds one (chunk, n, 6) output plus one (chunk, half, 6)
        # noise buffer instead of two copies of every part.
        daughters = torch.empty(
            (number_parents, upsampling_factor, 6),
            dtype=parents.dtype,
            device=torch_device,
        )
        if half > 0:
            noise = torch.randn(
                (number_parents, half, 6),
                generator=generator,
                dtype=parents.dtype,
                device=torch_device,
            )
            noise *= (sigma * bandwidth)[:, None, :]  # delta, (chunk, half, 6)
            centre = parents[:, None, :]
            torch.add(centre, noise, out=daughters[:, :half])
            torch.sub(centre, noise, out=daughters[:, half : 2 * half])
            del noise
        if upsampling_factor % 2 == 1:
            daughters[:, -1] = parents

        return daughters.reshape(number_parents * upsampling_factor, 6)

    def build_dataframe(
        self, position: np.ndarray, momentum: np.ndarray, weight: np.ndarray
    ) -> pd.DataFrame:
        columns = {}
        for i, name in enumerate(POSITION_COLUMNS):
            columns[name] = position[:, i]
        for i, name in enumerate(MOMENTUM_COLUMNS):
            columns[name] = momentum[:, i]
        columns[self.weight_column] = weight
        if "kinetic_energy_mev" in self.df.columns:
            columns["kinetic_energy_mev"] = (
                np.sqrt(np.sum(momentum**2, axis=1) + self.mass**2) - self.mass
            )

        upsampled_df = pd.DataFrame(columns, dtype=np.float32)
        return upsampled_df[[name for name in self.df.columns if name in columns]]

    def log_upsample_statistics(
        self,
        number_initial: int,
        upsampling_factor: int,
        weight: np.ndarray,
        momentum: np.ndarray,
        total_weight_before: float,
        total_momentum_before: np.ndarray,
        total_energy_before: float,
    ):
        number_final = weight.shape[0]
        weight = weight.astype(np.float64)
        momentum = momentum.astype(np.float64)
        energy = np.sqrt(np.sum(momentum**2, axis=1) + self.mass**2)
        logger.info(
            "Kernel upsampling: %d -> %d macroparticles (factor %d).\n",
            number_initial,
            number_final,
            upsampling_factor,
        )
        logger.info(
            "Relative conservation errors: weight %.2e, momentum %.2e, energy %.2e.\n",
            abs(weight.sum() - total_weight_before) / total_weight_before,
            np.linalg.norm(weight @ momentum - total_momentum_before)
            / max(np.linalg.norm(total_momentum_before), np.finfo(float).eps),
            abs(weight @ energy - total_energy_before) / total_energy_before,
        )
