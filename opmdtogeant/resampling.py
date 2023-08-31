"""
This module contains various particle resampling strategies for particle in cell data.
"""
import numpy as np
import pandas as pd


class ParticleResampler:
    def __init__(self, dataframe: pd.DataFrame, weight_column: str = "weights"):
        self.dataframe = dataframe.copy()
        self.weight_column = weight_column

    def set_weights_to(self, new_weight: int = 1) -> pd.DataFrame:
        self.dataframe[self.weight_column] = new_weight

        return self

    def random_weights(self) -> pd.DataFrame:
        min_weight = self.dataframe[self.weight_column].min()
        max_weight = self.dataframe[self.weight_column].max()
        random_generator = np.random.default_rng(seed=42)
        self.dataframe[self.weight_column] = random_generator.uniform(
            min_weight, max_weight, size=self.dataframe.shape[0]
        )

        return self

    def simple_thinning(self, number_of_remaining_particles: int) -> pd.DataFrame:
        random_generator = np.random.default_rng(seed=42)
        number_of_initial_particles = self.dataframe.shape[0]

        # Generate random indices for deletion
        delete_indices = random_generator.choice(
            number_of_initial_particles,
            size=number_of_initial_particles - number_of_remaining_particles,
            replace=False,
        )

        # Delete particles and weights at the selected indices
        self.dataframe = self.dataframe.drop(delete_indices, inplace=True)

        # Calculate new weight coefficient and update weights
        weight_factor = number_of_initial_particles / number_of_remaining_particles
        self.dataframe[self.weight_column] *= weight_factor

        return self

    def global_leveling_thinning(self, k: float = 2.0) -> pd.DataFrame:
        average_weight = self.dataframe[self.weight_column].mean()
        threshold_weight = k * average_weight

        # Generate random numbers for each particle
        random_generator = np.random.default_rng(seed=42)
        random_numbers = random_generator.uniform(
            0.0, 1.0, size=self.dataframe.shape[0]
        )

        # Create a mask for particles to be deleted
        deletion_mask = (self.dataframe[self.weight_column] < threshold_weight) & (
            random_numbers > (self.dataframe[self.weight_column] / threshold_weight)
        )

        # Delete particles
        self.dataframe.drop(self.dataframe[deletion_mask].index, inplace=True)

        # Update weights for the remaining particles
        self.dataframe[self.weight_column] = self.dataframe[self.weight_column].where(
            self.dataframe[self.weight_column] >= threshold_weight, threshold_weight
        )

        return self

    def finalize(self) -> pd.DataFrame:
        return self.dataframe
