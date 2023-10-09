"""
This module contains various particle resampling strategies for particle in cell data.
"""
import numpy as np
import pandas as pd

from .log import logger
from .utils import dataset_info
from .reader import DataFrameUpdater


class ParticleResampler:
    def __init__(self, df: pd.DataFrame, weight_column: str = "weights"):
        self._df = df.copy()
        self.weight_column = weight_column
        self.updater = DataFrameUpdater(self)

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
            logger.info("Multiplicative factor: %s\n", self.df[self.weight_column][0])
            self.df[self.weight_column] = new_weight

        return self

    def random_weights(self) -> pd.DataFrame:
        min_weight = self.df[self.weight_column].min()
        max_weight = self.df[self.weight_column].max()
        random_generator = np.random.default_rng(seed=42)
        self.df[self.weight_column] = random_generator.uniform(
            min_weight, max_weight, size=self.df.shape[0]
        )

        return self

    def simple_thinning(self, number_of_remaining_macroparticles: int) -> pd.DataFrame:
        number_of_remaining_macroparticles = int(number_of_remaining_macroparticles)

        random_generator = np.random.default_rng(seed=42)
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
        random_generator = np.random.default_rng(seed=42)
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

    def repeat_and_perturb(self, percentage: float = 0.001) -> pd.DataFrame:
        """
        Repeat each row based on the 'weights' column, set all 'weights' to 1,
        and add a small random value to the position and momentum columns.
        """
        random_generator = np.random.default_rng(seed=42)

        # Drop energy column
        energy_mev_dropped = False
        if "energy_mev" in self.df.columns:
            self.df.drop(columns=["energy_mev"], inplace=True)
            energy_mev_dropped = True

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
        if energy_mev_dropped:
            self.updater.add_energy_column()

        return self

    def finalize(self) -> pd.DataFrame:
        logger.info("Final dataset to be exported.\n")
        dataset_info(self.df)
        return self.df
