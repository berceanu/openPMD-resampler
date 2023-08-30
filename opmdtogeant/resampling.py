"""
This module contains various particle resampling strategies for particle in cell data.
"""
import numpy as np
import pandas as pd


class ParticleResampler:
    def __init__(self, dataframe: pd.DataFrame, weight_column: str = "weights"):
        self.dataframe = dataframe.copy()
        self.weight_column = weight_column

    def set_weights_to(self, new_weight: int = 1):
        self.dataframe[self.weight_column] = new_weight
        return self.dataframe

    def simple_thinning(self, number_of_remaining_particles: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed=42)
        number_of_initial_particles = len(self.dataframe)

        # Generate random indices for deletion
        delete_indices = rng.choice(
            number_of_initial_particles,
            size=number_of_initial_particles - number_of_remaining_particles,
            replace=False,
        )

        # Delete particles and weights at the selected indices
        self.dataframe = self.dataframe.drop(delete_indices)

        # Calculate new weight coefficient and update weights
        weight_factor = number_of_initial_particles / number_of_remaining_particles
        print(f"weight factor = {weight_factor}")
        self.dataframe[self.weight_column] *= weight_factor

        return self.dataframe

    def global_leveling_thinning(self, leveling_factor: float) -> pd.DataFrame:
        # Implement the global leveling thinning algorithm here
        # This is a placeholder and should be replaced with the actual implementation
        return self.dataframe
