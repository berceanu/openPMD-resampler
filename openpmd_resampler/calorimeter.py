"""
Generate macroparticles from a PIConGPU particle calorimeter and a density mesh.

The rest of the package goes particles -> fewer (or more) particles. This module
creates particles from binned distributions instead. The momentum distribution
comes from the PIConGPU particle calorimeter plugin, and the positions come from
a species density mesh written by the openPMD plugin at the same iteration.

Calorimeter format, taken from the PIConGPU kernel
(plugins/particleCalorimeter/ParticleCalorimeterFunctors.hpp and
ParticleCalorimeter.x.cpp):

    * mesh "calorimeter", shape [N_energy, N_pitch, N_yaw]; value * unitSI is the
      total kinetic energy in joule of the real particles in that bin, so the
      real particle count of a bin is its energy divided by the mean kinetic
      energy per particle in it;
    * energy bin 0 collects everything below minEnergy, bin N_energy - 1
      everything above maxEnergy; inner bin k in [1, N_energy - 2] covers the
      fraction [(k - 1) / (N_energy - 2), k / (N_energy - 2)) of
      [minEnergy, maxEnergy], in log10 when logScale is set, linearly otherwise
      (PIConGPU's python getEnergy() returns logspace(min, max, N), which does
      not match this layout);
    * in the calorimeter frame, yaw = atan2(d_x, d_y) and pitch = asin(d_z), with
      bin k covering max * (2 k / N - 1) to max * (2 (k + 1) / N - 1), where
      maxYaw[deg] / maxPitch[deg] are half the opening angles;
    * the calorimeter frame is rotated so that its +y axis points to
      (posYaw, posPitch), with zero roll.

Neither file links positions to momenta: the calorimeter has no positions and
the density mesh has no momenta. Positions and momenta are therefore sampled
independently, so position-momentum correlations are not reproduced.

Within a bin, energies are uniform in log10(E) (log bins) or in E (linear bins),
yaw is uniform and pitch is uniform in sin(pitch), i.e. uniform in solid angle.
The real count of a bin uses the mean energy of that within-bin law, so the
total kinetic energy is conserved in expectation. Positions are uniform within
a density cell drawn with probability proportional to its density. Like
reader.py does for PIConGPU data, the y and z axes are swapped at the end, so
the calorimeter's default forward direction (+y in PIConGPU) becomes +z.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from .log import logger
from .units import constants

SEED = 77125

POSITION_COLUMNS = ["position_x_m", "position_y_m", "position_z_m"]
MOMENTUM_COLUMNS = ["momentum_x_mev_c", "momentum_y_mev_c", "momentum_z_mev_c"]

JOULE_PER_KEV = 1e3 * constants.joule_to_ev


@dataclass
class CalorimeterData:
    """Calorimeter energies in joule, shape [N_energy, N_pitch, N_yaw]."""

    energy_joule: np.ndarray
    min_energy_kev: float
    max_energy_kev: float
    log_scale: bool
    max_yaw_deg: float
    max_pitch_deg: float
    pos_yaw_deg: float = 0.0
    pos_pitch_deg: float = 0.0
    iteration: Optional[int] = None


@dataclass
class DensityGrid:
    """
    A density mesh, used as a relative probability density for positions.

    values is stored in the mesh's own axis order, axis_order names the
    simulation axis ("x", "y" or "z") of each array dimension, and
    cell_size_m / origin_m map each name to the cell edge length and the
    global offset of the grid in metres.
    """

    values: np.ndarray
    axis_order: Tuple[str, ...]
    cell_size_m: dict
    origin_m: dict
    iteration: Optional[int] = None


def _single_iteration(series, path):
    iteration_numbers = tuple(series.iterations)
    if len(iteration_numbers) != 1:
        raise ValueError(
            f"{path} should contain exactly one iteration, found {iteration_numbers}."
        )
    return iteration_numbers[0], series.iterations[iteration_numbers[0]]


def read_calorimeter(path: Union[str, Path]) -> CalorimeterData:
    """Read a PIConGPU calorimeter file (one iteration, with energy binning)."""
    import openpmd_api as io

    series = io.Series(str(path), io.Access.read_only)
    iteration_number, iteration = _single_iteration(series, path)
    if "calorimeter" not in iteration.meshes:
        raise ValueError(
            f"{path} has no 'calorimeter' mesh; found {list(iteration.meshes)}."
        )
    component = iteration.meshes["calorimeter"][io.Mesh_Record_Component.SCALAR]
    if len(component.shape) != 3:
        raise ValueError(
            f"{path} has no energy binning (shape {component.shape}); particle"
            " energies cannot be recovered. Rerun with --<sp>_calorimeter.numBinsEnergy > 1."
        )
    attribute = component.get_attribute
    data = component.load_chunk()
    series.flush()
    energy_joule = data.astype(np.float64) * attribute("unitSI")
    calorimeter = CalorimeterData(
        energy_joule=energy_joule,
        min_energy_kev=float(attribute("minEnergy[keV]")),
        max_energy_kev=float(attribute("maxEnergy[keV]")),
        log_scale=bool(attribute("logScale")),
        max_yaw_deg=float(attribute("maxYaw[deg]")),
        max_pitch_deg=float(attribute("maxPitch[deg]")),
        pos_yaw_deg=float(attribute("posYaw[deg]")),
        pos_pitch_deg=float(attribute("posPitch[deg]")),
        iteration=int(iteration_number),
    )
    series.close()
    logger.info(
        "Read calorimeter %s: iteration %d, shape %s (energy, pitch, yaw), %s energy bins"
        " from %.4g to %.4g keV, yaw +-%.4g deg, pitch +-%.4g deg around (%.4g, %.4g) deg.\n",
        path,
        calorimeter.iteration,
        energy_joule.shape,
        "log" if calorimeter.log_scale else "linear",
        calorimeter.min_energy_kev,
        calorimeter.max_energy_kev,
        calorimeter.max_yaw_deg,
        calorimeter.max_pitch_deg,
        calorimeter.pos_yaw_deg,
        calorimeter.pos_pitch_deg,
    )
    return calorimeter


def read_density(path: Union[str, Path], mesh_name: str) -> DensityGrid:
    """Read a scalar density mesh (e.g. 'H_all_density') from an openPMD file."""
    import openpmd_api as io

    series = io.Series(str(path), io.Access.read_only)
    iteration_number, iteration = _single_iteration(series, path)
    available = list(iteration.meshes)
    if mesh_name not in available:
        raise ValueError(
            f"Density mesh '{mesh_name}' not available in {path}. Detected meshes: {available}"
        )
    mesh = iteration.meshes[mesh_name]
    component = mesh[io.Mesh_Record_Component.SCALAR]
    values = component.load_chunk()
    series.flush()

    axis_order = tuple(mesh.axis_labels)
    if len(axis_order) != 3 or sorted(axis_order) != ["x", "y", "z"]:
        raise ValueError(f"Expected a 3-D x/y/z mesh, got axis labels {axis_order}.")
    unit = mesh.grid_unit_SI
    cell_size_m = {a: s * unit for a, s in zip(axis_order, mesh.grid_spacing)}
    origin_m = {a: o * unit for a, o in zip(axis_order, mesh.grid_global_offset)}
    series.close()
    logger.info(
        "Read density mesh '%s' from %s: iteration %d, shape %s (%s), cell size %s m.\n",
        mesh_name,
        path,
        iteration_number,
        values.shape,
        ", ".join(axis_order),
        ", ".join(f"{a}={cell_size_m[a]:.4g}" for a in "xyz"),
    )
    return DensityGrid(
        values=values,
        axis_order=axis_order,
        cell_size_m=cell_size_m,
        origin_m=origin_m,
        iteration=int(iteration_number),
    )


def calorimeter_frame(pos_yaw_deg: float, pos_pitch_deg: float) -> np.ndarray:
    """
    Rows are the calorimeter frame's x, y, z axes in simulation coordinates,
    as in ParticleCalorimeter.x.cpp, so d_cal = R @ d_sim and d_sim = R.T @ d_cal.
    """
    yaw = np.deg2rad(pos_yaw_deg)
    pitch = np.deg2rad(pos_pitch_deg)
    frame_y = np.array([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    if abs(frame_y[2]) == 1.0:
        frame_x = np.array([1.0, 0.0, 0.0])
    else:
        frame_x = np.cross(np.array([0.0, 0.0, -1.0]), frame_y)
        frame_x /= np.linalg.norm(frame_x)
    frame_z = np.cross(frame_x, frame_y)
    return np.stack([frame_x, frame_y, frame_z])


def _inverse_cdf_draw(rng, cumulative: np.ndarray, size: int) -> np.ndarray:
    """Indices drawn with probability proportional to the increments of cumulative."""
    draws = rng.random(size) * cumulative[-1]
    return np.minimum(np.searchsorted(cumulative, draws, side="right"), cumulative.size - 1)


class CalorimeterSampler:
    def __init__(
        self,
        calorimeter: CalorimeterData,
        density: Optional[DensityGrid],
        mass_mev_c2: float,
        energy_range_mev: Optional[Tuple[float, float]] = None,
        seed: int = SEED,
    ):
        """
        calorimeter: the momentum distribution.
        density: the position distribution; None puts every particle at the origin.
        mass_mev_c2: species rest mass in MeV/c^2 (0.0 for photons).
        energy_range_mev: optional (min, max) kinetic energy window in MeV;
            only energy bins entirely inside it are used.
        """
        if mass_mev_c2 < 0.0:
            raise ValueError("mass_mev_c2 must be non-negative.")
        self.calorimeter = calorimeter
        self.density = density
        self.mass = mass_mev_c2
        self.energy_range_mev = energy_range_mev
        self.rng = np.random.default_rng(seed)
        if (
            density is not None
            and calorimeter.iteration is not None
            and density.iteration is not None
            and calorimeter.iteration != density.iteration
        ):
            logger.warning(
                "Calorimeter iteration %d differs from density iteration %d.\n",
                calorimeter.iteration,
                density.iteration,
            )

    def energy_edges_kev(self) -> np.ndarray:
        """Edges of the inner energy bins 1 .. N_energy - 2 (N_energy - 1 values)."""
        cal = self.calorimeter
        number_inner = cal.energy_joule.shape[0] - 2
        if number_inner < 1:
            raise ValueError("The calorimeter needs at least 3 energy bins.")
        if cal.log_scale:
            return np.logspace(
                np.log10(cal.min_energy_kev), np.log10(cal.max_energy_kev), number_inner + 1
            )
        return np.linspace(cal.min_energy_kev, cal.max_energy_kev, number_inner + 1)

    def bin_table(self) -> dict:
        """
        Every non-empty inner bin with its edges and real particle count.

        Returns a dict of 1-D arrays: energy_lo_kev, energy_hi_kev, yaw_lo, yaw_hi,
        pitch_lo, pitch_hi (radians, calorimeter frame), energy_joule, n_real.
        """
        cal = self.calorimeter
        energy = cal.energy_joule
        total = energy.sum()
        if total <= 0.0:
            raise ValueError("The calorimeter is empty.")
        logger.info(
            "Calorimeter total kinetic energy %.6e J; underflow bin (< %.4g keV) %.2f%%,"
            " overflow bin (> %.4g keV) %.2f%%, both excluded.\n",
            total,
            cal.min_energy_kev,
            100.0 * energy[0].sum() / total,
            cal.max_energy_kev,
            100.0 * energy[-1].sum() / total,
        )

        edges = self.energy_edges_kev()
        inner = energy[1:-1]
        keep_energy = np.ones(inner.shape[0], dtype=bool)
        if self.energy_range_mev is not None:
            low, high = self.energy_range_mev
            keep_energy = (edges[:-1] >= 1e3 * low) & (edges[1:] <= 1e3 * high)
            logger.info(
                "Energy window %.4g - %.4g MeV keeps %d of %d energy bins (%.2f%% of the"
                " in-range energy).\n",
                low,
                high,
                keep_energy.sum(),
                keep_energy.size,
                100.0 * inner[keep_energy].sum() / max(inner.sum(), np.finfo(float).tiny),
            )
        inner = np.where(keep_energy[:, None, None], inner, 0.0)

        energy_index, pitch_index, yaw_index = np.nonzero(inner > 0.0)
        if energy_index.size == 0:
            raise ValueError("No non-empty calorimeter bins left to sample from.")
        energy_lo = edges[energy_index]
        energy_hi = edges[energy_index + 1]
        if cal.log_scale:
            mean_energy_kev = (energy_hi - energy_lo) / np.log(energy_hi / energy_lo)
        else:
            mean_energy_kev = 0.5 * (energy_lo + energy_hi)
        bin_energy = inner[energy_index, pitch_index, yaw_index]

        number_pitch, number_yaw = energy.shape[1:]
        max_yaw = np.deg2rad(cal.max_yaw_deg)
        max_pitch = np.deg2rad(cal.max_pitch_deg)
        return {
            "energy_lo_kev": energy_lo,
            "energy_hi_kev": energy_hi,
            "yaw_lo": max_yaw * (2.0 * yaw_index / number_yaw - 1.0),
            "yaw_hi": max_yaw * (2.0 * (yaw_index + 1) / number_yaw - 1.0),
            "pitch_lo": max_pitch * (2.0 * pitch_index / number_pitch - 1.0),
            "pitch_hi": max_pitch * (2.0 * (pitch_index + 1) / number_pitch - 1.0),
            "energy_joule": bin_energy,
            "n_real": bin_energy / (mean_energy_kev * JOULE_PER_KEV),
        }

    def sample(
        self,
        mode: str = "random",
        num_macroparticles: int = 1_000_000,
        particles_per_bin: int = 1,
    ) -> pd.DataFrame:
        """
        mode "random": num_macroparticles drawn from the calorimeter treated as a
            probability distribution, all with weight N_real / num_macroparticles.
        mode "per_bin": particles_per_bin macroparticles in every non-empty bin,
            each with weight n_real(bin) / particles_per_bin. This reproduces the
            histogram exactly, including sparse tails, at the cost of
            non-uniform weights.
        """
        table = self.bin_table()
        n_real = table["n_real"]
        total_real = n_real.sum()

        if mode == "random":
            number = int(num_macroparticles)
            if number < 1:
                raise ValueError("num_macroparticles must be at least 1.")
            bins = _inverse_cdf_draw(self.rng, np.cumsum(n_real), number)
            weights = np.full(number, total_real / number)
        elif mode == "per_bin":
            per_bin = int(particles_per_bin)
            if per_bin < 1:
                raise ValueError("particles_per_bin must be at least 1.")
            bins = np.repeat(np.arange(n_real.size), per_bin)
            weights = n_real[bins] / per_bin
        else:
            raise ValueError(f"Unknown calorimeter sampling mode '{mode}'.")
        number = bins.size

        kinetic_mev = 1e-3 * self._sample_energy_kev(table, bins)
        direction = self._sample_direction(table, bins)
        momentum_norm = np.sqrt(kinetic_mev * (kinetic_mev + 2.0 * self.mass))
        momentum = direction * momentum_norm[:, None]
        position = self._sample_positions(number)

        # Same convention as OpenPMDLoader.swap_yz_axes: propagation along z.
        position = position[:, [0, 2, 1]]
        momentum = momentum[:, [0, 2, 1]]

        energy_before = table["energy_joule"].sum()
        energy_after = (weights * kinetic_mev).sum() * 1e3 * JOULE_PER_KEV
        logger.info(
            "Calorimeter sampling (%s): %d non-empty bins, %.6e real particles,"
            " %d macroparticles (weights %.4g - %.4g).\n",
            mode,
            n_real.size,
            total_real,
            number,
            weights.min(),
            weights.max(),
        )
        logger.info(
            "Kinetic energy: calorimeter %.6e J, sampled %.6e J, relative error %.2e.\n",
            energy_before,
            energy_after,
            abs(energy_after - energy_before) / energy_before,
        )

        columns = {}
        for i, name in enumerate(POSITION_COLUMNS):
            columns[name] = position[:, i]
        for i, name in enumerate(MOMENTUM_COLUMNS):
            columns[name] = momentum[:, i]
        columns["weights"] = weights
        columns["kinetic_energy_mev"] = kinetic_mev
        return pd.DataFrame(columns, dtype=np.float32)

    def _sample_energy_kev(self, table: dict, bins: np.ndarray) -> np.ndarray:
        low = table["energy_lo_kev"][bins]
        high = table["energy_hi_kev"][bins]
        uniform = self.rng.random(bins.size)
        if self.calorimeter.log_scale:
            return low * (high / low) ** uniform
        return low + uniform * (high - low)

    def _sample_direction(self, table: dict, bins: np.ndarray) -> np.ndarray:
        """Unit directions in simulation coordinates (before the y/z swap)."""
        yaw_lo, yaw_hi = table["yaw_lo"][bins], table["yaw_hi"][bins]
        sin_lo, sin_hi = np.sin(table["pitch_lo"][bins]), np.sin(table["pitch_hi"][bins])
        yaw = yaw_lo + self.rng.random(bins.size) * (yaw_hi - yaw_lo)
        sin_pitch = sin_lo + self.rng.random(bins.size) * (sin_hi - sin_lo)
        cos_pitch = np.sqrt(np.clip(1.0 - sin_pitch**2, 0.0, None))
        direction_calorimeter = np.stack(
            [cos_pitch * np.sin(yaw), cos_pitch * np.cos(yaw), sin_pitch], axis=1
        )
        frame = calorimeter_frame(self.calorimeter.pos_yaw_deg, self.calorimeter.pos_pitch_deg)
        return direction_calorimeter @ frame

    def _sample_positions(self, number: int) -> np.ndarray:
        """Positions (x, y, z) in metres in simulation coordinates."""
        if self.density is None:
            logger.info("No density given; all particles are placed at the origin.\n")
            return np.zeros((number, 3))
        grid = self.density
        flat = grid.values.reshape(-1)
        occupied = np.flatnonzero(flat > 0)
        if occupied.size == 0:
            raise ValueError("The density mesh has no positive cells.")
        cumulative = np.cumsum(flat[occupied], dtype=np.float64)
        cells = occupied[_inverse_cdf_draw(self.rng, cumulative, number)]
        number_occupied = occupied.size
        del cumulative, occupied
        index = np.unravel_index(cells, grid.values.shape)
        position = np.empty((number, 3))
        for dimension, axis in enumerate(grid.axis_order):
            column = "xyz".index(axis)
            position[:, column] = grid.origin_m[axis] + (
                index[dimension] + self.rng.random(number)
            ) * grid.cell_size_m[axis]
        logger.info(
            "Positions drawn from %d occupied density cells.\n", number_occupied
        )
        return position


def particles_from_calorimeter(
    calorimeter_path: Union[str, Path],
    density_path: Optional[Union[str, Path]],
    density_mesh: str,
    mass_mev_c2: float,
    mode: str = "random",
    num_macroparticles: int = 1_000_000,
    particles_per_bin: int = 1,
    energy_range_mev: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Read both files and sample a DataFrame with the package's column contract."""
    calorimeter = read_calorimeter(calorimeter_path)
    density = read_density(density_path, density_mesh) if density_path is not None else None
    sampler = CalorimeterSampler(
        calorimeter, density, mass_mev_c2, energy_range_mev=energy_range_mev
    )
    return sampler.sample(
        mode=mode,
        num_macroparticles=num_macroparticles,
        particles_per_bin=particles_per_bin,
    )
