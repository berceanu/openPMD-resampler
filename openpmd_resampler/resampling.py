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
        self.df = df.copy()
        self.weight_column = weight_column
        self.updater = DataFrameUpdater(self.df)

    def set_weights_to(self, new_weight: int = 1) -> pd.DataFrame:
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

    def global_leveling_thinning(self, k: float = 2.0) -> pd.DataFrame:
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
        self.df.drop(self.df[deletion_mask].index, inplace=True)

        # Update weights for the remaining particles
        self.df[self.weight_column] = self.df[self.weight_column].where(
            self.df[self.weight_column] >= threshold_weight, threshold_weight
        )

        return self

    def repeat_and_perturb(self, epsilon_ratio: float = 0.01) -> pd.DataFrame:
        """
        Repeat each row based on the 'weights' column, set all 'weights' to 1,
        add a small random value between -epsilon and epsilon to the position and momentum columns.
        """
        random_generator = np.random.default_rng(seed=42)

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

        # Add small epsilon to each column
        for col in cols:
            epsilon = (
                self.df[col] * epsilon_ratio
            )  # epsilon is a percentage of the value in each cell
            self.df[col] += random_generator.uniform(-epsilon, epsilon)

        # Reset the index
        self.df.reset_index(drop=True, inplace=True)

        if energy_mev_dropped:
            self.updater.add_energy_column()

        return self

    def finalize(self) -> pd.DataFrame:
        logger.info("Reducing number of particles.\n")
        dataset_info(self.df)
        return self.df
