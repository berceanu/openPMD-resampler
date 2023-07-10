from typing import Dict, Tuple

import numpy as np
import openpmd_api as io
import pandas as pd
import sparklines


class HDF5Reader:
    def __init__(self, file_path: str):
        """
        Constructor for HDF5Reader.

        :param file_path: path to the HDF5 file
        """
        self.file_path = file_path

    def _get_iteration(self, series: io.Series) -> io.Iteration:
        """
        Gets the iteration number of the series.

        :param series: the HDF5 series
        :return: the iteration
        """
        it_numbers = tuple(series.iterations)
        if len(it_numbers) != 1:
            raise ValueError(f"{self.file_path} should contain exactly one iteration.")

        it_number = it_numbers[0]
        print(f"{self.file_path} contains iteration {it_number}.")
        return series.iterations[it_number]

    def _get_data_chunk_and_units(
        self, data: io.ParticleSpecies, component: str
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Gets data chunk, unit and unit dimension for the given component.

        :param data: particle species data
        :param component: the component ("x", "y", "z")
        :return: data chunk, unit and unit dimension
        """
        unit_dimension = data.unit_dimension
        data_component = data[component]
        data_unit = data_component.unit_SI
        data_chunk = data_component.load_chunk()
        return data_chunk, data_unit, unit_dimension

    def _check_dimension(self, actual: np.ndarray, expected: np.ndarray):
        """
        Asserts that actual and expected dimensions are close.

        :param actual: actual dimension
        :param expected: expected dimension
        """
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)

    def read_file(self) -> pd.DataFrame:
        """
        Reads the HDF5 file and returns a DataFrame with position, momentum and weight data.

        :return: DataFrame with position, momentum and weight data
        """
        series = io.Series(self.file_path, io.Access.read_only)

        code_is_picongpu = series.software == "PIConGPU"

        it = self._get_iteration(series)
        electrons = it.particles["e_all"]

        data_dict = {}
        units_dict = {}
        unit_dims_dict = {}

        for attribute in ["position", "momentum"]:
            for component in ["x", "y", "z"]:
                (
                    data_dict[f"{attribute}_{component}"],
                    units_dict[f"{attribute}_{component}"],
                    unit_dims_dict[f"{attribute}_{component}"],
                ) = self._get_data_chunk_and_units(electrons[attribute], component)

        (
            data_dict["weights"],
            units_dict["weights"],
            unit_dims_dict["weights"],
        ) = self._get_data_chunk_and_units(
            electrons["weighting"], io.Record_Component.SCALAR
        )

        series.flush()
        del series

        expected_dims = {
            "position": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "momentum": np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
            "weights": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }

        for key, unit_dim in unit_dims_dict.items():
            attribute = key.split("_")[0]
            self._check_dimension(unit_dim, expected_dims[attribute])

        df = pd.DataFrame(data_dict)

        for column, unit in units_dict.items():
            df[column] *= unit

        if code_is_picongpu:  # swap y and z axes
            for attribute in ["position", "momentum"]:
                df[[f"{attribute}_y", f"{attribute}_z"]] = df[
                    [f"{attribute}_z", f"{attribute}_y"]
                ]

        print("The laser is propagating along the z direction.")
        print(
            f"The dataset contains {df.shape[0]:,} macroparticles, corresponding to {int(df['weights'].sum()):,} 'real' electrons."
        )
        print("Descriptive statistics of the dataset:")
        print(df.describe())

        spark_hist = HDF5Reader.create_weight_histogram(df, num_bins=10, num_lines=3)
        print("Histogram of the weights column:")
        print(spark_hist)

        return df

    @staticmethod
    def create_weight_histogram(
        df: pd.DataFrame, num_bins: int = 30, num_lines: int = 2
    ) -> str:
        """
        Creates a histogram of the weights column of the dataframe with a given number of bins
        and returns a sparklines string representation of it with a given number of lines.

        :param df: DataFrame with position, momentum and weight data
        :param num_bins: number of bins for the histogram, default is 30
        :param num_lines: number of lines for the sparklines representation, default is 2
        :return: a string with sparklines representation of the histogram
        """
        hist, bin_edges = np.histogram(df["weights"], bins=num_bins)
        histogram_sparklines = "\n".join(
            sparklines.sparklines(hist, num_lines=num_lines)
        )
        return histogram_sparklines
