"""
This module contains various particle resampling strategies for particle in cell data.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from .log import logger
from .units import constants
from .utils import dataset_info
from .reader import DataFrameUpdater
from .vranic import VranicMerger
from .voronoi import VoronoiMerger
from .upsampling import KernelUpsampler


class ParticleResampler:
    def __init__(
        self,
        df: pd.DataFrame,
        weight_column: str = "weights",
        particle_species_mass: float = 1.0,
    ):
        self._df = df.copy()
        self.weight_column = weight_column
        self.particle_species_mass = particle_species_mass
        self.updater = DataFrameUpdater(self, particle_species_mass)

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    def set_weights_to(self, new_weight: int = 1) -> pd.DataFrame:
        if new_weight == 1 and self.df[self.weight_column].nunique() != 1:
            raise ValueError("Not all weights are equal. Setting them to 1 might not be a good idea.")
        else:
            logger.info("Multiplicative factor for obtaining original charge: %.2f\n", self.df[self.weight_column].iloc[0])
            self.df[self.weight_column] = new_weight

        return self

    def random_weights(self) -> pd.DataFrame:
        min_weight = self.df[self.weight_column].min()
        max_weight = self.df[self.weight_column].max()
        random_generator = np.random.default_rng(seed=77125)
        self.df[self.weight_column] = random_generator.uniform(
            min_weight, max_weight, size=self.df.shape[0]
        )

        return self

    def simple_thinning(self, number_of_remaining_macroparticles: int) -> pd.DataFrame:
        number_of_remaining_macroparticles = int(number_of_remaining_macroparticles)

        random_generator = np.random.default_rng(seed=77125)
        number_of_initial_macroparticles = self.df.shape[0]

        # Generate random indices for deletion
        delete_indices = random_generator.choice(
            number_of_initial_macroparticles,
            size=number_of_initial_macroparticles - number_of_remaining_macroparticles,
            replace=False,
        )

        # Delete particles and weights at the selected indices
        self.df.drop(delete_indices, inplace=True)

        # Calculate new weight coefficient and update weights
        weight_factor = (
            number_of_initial_macroparticles / number_of_remaining_macroparticles
        )
        self.df[self.weight_column] *= weight_factor

        return self

    def drop_rows_using_mask(self, deletion_mask):
        self.df.loc[deletion_mask] = None
        self.df.dropna(inplace=True)

    def global_leveling_thinning(self, k: float = 2.0) -> pd.DataFrame:
        """
        If the initial number of macroparticles is N, the number after
        thinning will be roughly N/k.
        """
        average_weight = self.df[self.weight_column].mean()
        threshold_weight = k * average_weight

        # Generate random numbers for each particle
        random_generator = np.random.default_rng(seed=77125)
        random_numbers = random_generator.uniform(0.0, 1.0, size=self.df.shape[0])

        # Create a mask for particles to be deleted
        deletion_mask = (self.df[self.weight_column] < threshold_weight) & (
            random_numbers > (self.df[self.weight_column] / threshold_weight)
        )

        # Delete particles
        self.drop_rows_using_mask(deletion_mask)

        # Update weights for the remaining particles
        self.df[self.weight_column] = self.df[self.weight_column].where(
            self.df[self.weight_column] >= threshold_weight, threshold_weight
        )

        logger.info("Dataset after thinning.\n")
        dataset_info(self.df)

        return self

    def vranic_merging(
        self,
        spatial_bins: Tuple[int, int, int] = (16, 16, 16),
        momentum_bins: Tuple[int, int, int] = (16, 16, 16),
        momentum_coordinates: str = "spherical",
        min_packet_size: int = 4,
        max_packet_size: int = 4,
        log_scale: bool = False,
        device: str = None,
    ) -> pd.DataFrame:
        """
        Merge macroparticles using the algorithm of Vranic et al.,
        Comput. Phys. Commun. 191 (2015) 65-73.

        Particles are binned into spatial cells and momentum cells, and each
        packet of at least min_packet_size particles sharing a cell is replaced
        by two macroparticles conserving total weight, momentum and energy.
        Coarser binning merges more aggressively.

        The particle mass is taken from the particle_species_mass given to the
        constructor (relative to the electron mass, 0.0 for photons).

        The merge runs on PyTorch tensors; device selects where ("cuda",
        "cuda:1", "cpu", ...). The default uses the GPU when one is available
        (NVIDIA CUDA and AMD ROCm builds both expose it as "cuda").
        """
        merger = VranicMerger(
            self.df,
            mass_mev_c2=self.particle_species_mass * constants.electron_mass_mev_c2,
            weight_column=self.weight_column,
        )
        self.df = merger.merge(
            spatial_bins=spatial_bins,
            momentum_bins=momentum_bins,
            momentum_coordinates=momentum_coordinates,
            min_packet_size=min_packet_size,
            max_packet_size=max_packet_size,
            log_scale=log_scale,
            device=device,
        )

        logger.info("Dataset after Vranic merging.\n")
        dataset_info(self.df)

        return self

    def voronoi_merging(
        self,
        spatial_bins: Tuple[int, int, int] = (16, 16, 16),
        min_particles_to_merge: int = 8,
        pos_spread_threshold: float = 0.5,
        abs_mom_spread_threshold: float = -1.0,
        rel_mom_spread_threshold: float = -1.0,
        min_mean_energy_kev: float = 511.0,
        device: str = None,
    ) -> pd.DataFrame:
        """
        Merge macroparticles using the Voronoi algorithm of Luu, Tueckmantel
        and Pukhov, Comput. Phys. Commun. 202 (2016) 165-174, as implemented
        in the particleMerging plugin of PIConGPU.

        Particles are grouped into spatial cells that are subdivided
        recursively, first in position and then in momentum space, until the
        spread of every cluster falls below the given thresholds; each cluster
        of at least min_particles_to_merge particles is then replaced by a
        single macroparticle, conserving total weight and momentum exactly and
        energy up to the momentum spread threshold. Exactly one of
        abs_mom_spread_threshold (in m_e*c) and rel_mom_spread_threshold must
        be positive; pos_spread_threshold is in units of the initial spatial
        cell edge length and min_mean_energy_kev in keV.

        The particle mass is taken from the particle_species_mass given to the
        constructor (relative to the electron mass, 0.0 for photons).

        The merge runs on PyTorch tensors; device selects where ("cuda",
        "cuda:1", "cpu", ...). The default uses the GPU when one is available
        (NVIDIA CUDA and AMD ROCm builds both expose it as "cuda").
        """
        merger = VoronoiMerger(
            self.df,
            mass_mev_c2=self.particle_species_mass * constants.electron_mass_mev_c2,
            weight_column=self.weight_column,
        )
        self.df = merger.merge(
            spatial_bins=spatial_bins,
            min_particles_to_merge=min_particles_to_merge,
            pos_spread_threshold=pos_spread_threshold,
            abs_mom_spread_threshold=abs_mom_spread_threshold,
            rel_mom_spread_threshold=rel_mom_spread_threshold,
            min_mean_energy_kev=min_mean_energy_kev,
            device=device,
        )

        logger.info("Dataset after Voronoi merging.\n")
        dataset_info(self.df)

        return self

    def kernel_upsampling(
        self,
        upsampling_factor: int = 10,
        spatial_bins: Tuple[int, int, int] = (16, 16, 16),
        position_bandwidth: float = 0.1,
        momentum_bandwidth: float = 0.1,
        device: str = None,
    ) -> pd.DataFrame:
        """
        Upsample macroparticles by antithetic kernel splitting: the inverse of
        the mergers, for a dataset with too few macroparticles for good
        post-processing statistics.

        Each parent macroparticle is split into upsampling_factor daughters of
        weight w/n, scattered around the parent by Gaussian noise whose width is
        the local weighted phase-space spread (estimated per spatial cell) times
        the position and momentum bandwidths. The daughters come in antithetic
        pairs (+delta, -delta), so total weight and total momentum are conserved
        exactly per parent; total energy is conserved only to second order in
        the bandwidth (the error is positive, shrinks with the bandwidth and is
        logged), because p(E) is concave and a nontrivial 1 -> many split of a
        massive particle cannot hold weight, momentum and energy at once.

        This is meant for post-processing statistics only, not for feeding the
        result back into a PIC simulation.

        The particle mass is taken from the particle_species_mass given to the
        constructor (relative to the electron mass, 0.0 for photons).

        The split runs on PyTorch tensors; device selects where ("cuda",
        "cuda:1", "cpu", ...). The default uses the GPU when one is available
        (NVIDIA CUDA and AMD ROCm builds both expose it as "cuda").
        """
        upsampler = KernelUpsampler(
            self.df,
            mass_mev_c2=self.particle_species_mass * constants.electron_mass_mev_c2,
            weight_column=self.weight_column,
        )
        self.df = upsampler.upsample(
            upsampling_factor=upsampling_factor,
            spatial_bins=spatial_bins,
            position_bandwidth=position_bandwidth,
            momentum_bandwidth=momentum_bandwidth,
            device=device,
        )

        logger.info("Dataset after kernel upsampling.\n")
        dataset_info(self.df)

        return self

    def repeat_and_perturb(self, percentage: float = 0.001) -> pd.DataFrame:
        """
        Repeat each row based on the 'weights' column, set all 'weights' to 1,
        and add a small random value to the position and momentum columns.
        """
        random_generator = np.random.default_rng(seed=77125)

        # Drop energy column
        kinetic_energy_mev_dropped = False
        if "kinetic_energy_mev" in self.df.columns:
            self.df.drop(columns=["kinetic_energy_mev"], inplace=True)
            kinetic_energy_mev_dropped = True

        # Repeat rows based on the 'weights' column
        self.df = self.df.loc[
            self.df.index.repeat(self.df[self.weight_column].astype(int))
        ]
        self.set_weights_to(1)

        # Get all columns except 'weights'
        cols = [col for col in self.df.columns if col != self.weight_column]

        # Compute mean and std for standardization
        mean = self.df.mean()
        std = self.df.std()

        # Standardize and add noise in-place
        for col in cols:
            self.df[col] = (self.df[col] - mean[col]) / std[col]
            epsilon = self.df[col].abs() * percentage
            self.df[col] += random_generator.normal(0, epsilon)
            self.df[col] = self.df[col] * std[col] + mean[col]

        self.df.reset_index(drop=True, inplace=True)

        # Recompute energy column
        if kinetic_energy_mev_dropped:
            self.updater.add_energy_column()

        return self

    def finalize(self) -> pd.DataFrame:
        logger.info("Final dataset to be exported.\n")
        dataset_info(self.df)
        return self.df
