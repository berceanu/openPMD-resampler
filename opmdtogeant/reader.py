from typing import Tuple

import numpy as np
import openpmd_api as io
import pandas as pd
import scipy.constants as const
from opmdtogeant.utils import dataset_info
from opmdtogeant.log import logger


# Constants
electron_charge_picocoulombs = const.elementary_charge * 1e12  # electron charge in pC
electron_mass_kg = const.m_e  # electron mass in kilograms
speed_of_light = const.c  # speed of light in m/s
joule_to_eV = const.electron_volt  # conversion factor from joules to electronvolts
eV_to_MeV = 1e6  # conversion factor from electronvolts to megaelectronvolts
meters_to_microns = 1e6  # conversion factor from meters to micrometers

# Electron mass in MeV/c^2
electron_mass_MeV_c2 = (electron_mass_kg * speed_of_light**2) / (
    joule_to_eV * eV_to_MeV
)

# Momentum in kg*m/s
momentum_kg_m_s = 1  # given
# Momentum in MeV/c
momentum_MeV_c = (momentum_kg_m_s * speed_of_light) / (joule_to_eV * eV_to_MeV)

MOMENTUM_CONVERSION_FACTOR = momentum_MeV_c
POSITION_CONVERSION_FACTOR = meters_to_microns
EXPECTED_DIMS = {
    "position": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "positionOffset": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "momentum": np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
    "weights": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
}
COMPONENTS = ["x", "y", "z"]


# TODO: Refactor this class - channge class/method/module names
class HDF5Reader:
    def __init__(self, file_path: str, particle_species_name: str = "e_all"):
        """Initialize HDF5Reader with the file path."""
        self.file_path = str(file_path)
        self.particle_species_name = particle_species_name

    def build_df(self) -> pd.DataFrame:
        """
        Reads the HDF5 file and returns a DataFrame with position, momentum and weight data.

        Example:
        reader = HDF5Reader("path_to_file.h5")
        df = reader.build_df()
        print(df)

        :return: DataFrame with position, momentum and weight data
        """
        series = self._open_series()
        self._software_name(series)
        swap_yz = series.software == "PIConGPU"

        iteration = self._get_iteration(series)
        electrons = iteration.particles[self.particle_species_name]

        data, units, dimensions = self._get_particle_data_and_units(electrons)

        self._check_dimensions(dimensions)

        # Flush chunks with actual data and delete series to free memory
        series.flush()

        macro_weighted = electrons["momentum"].get_attribute("macroWeighted")
        weighting_power = electrons["momentum"].get_attribute("weightingPower")
        if (macro_weighted == 1) and (weighting_power != 0):
            data = self._rescale_momenta(data, weighting_power)

        del series

        df = self._create_dataframe(data, units)

        df = self._update_data_frame(df, swap_axes=swap_yz)

        self._data_stats(df)

        return df

    def _open_series(self) -> io.Series:
        return io.Series(self.file_path, io.Access.read_only)

    def _software_name(self, series: io.Series):
        logger.info(
            "Data generated using %s %s.\n", series.software, series.software_version
        )

    def _get_iteration(self, series: io.Series) -> io.Iteration:
        iteration_numbers = tuple(series.iterations)
        if len(iteration_numbers) != 1:
            raise ValueError(f"{self.file_path} should contain exactly one iteration.")

        iteration_number = iteration_numbers[0]
        iteration = series.iterations[iteration_number]

        logger.info(
            "%s contains iteration %s, at %.2f ps.\n",
            self.file_path,
            iteration_number,
            iteration.time * iteration.time_unit_SI * 1e12,
        )
        return iteration

    def _get_particle_data_and_units(
        self, electrons: io.ParticleSpecies
    ) -> Tuple[dict, dict, dict]:
        data = {}
        units = {}
        dimensions = {}

        for attribute in ["position", "positionOffset", "momentum"]:
            for component in COMPONENTS:
                (
                    data[f"{attribute}_{component}"],
                    units[f"{attribute}_{component}"],
                    dimensions[f"{attribute}_{component}"],
                ) = self._get_component_data_and_units(electrons[attribute], component)

        (
            data["weights"],
            units["weights"],
            dimensions["weights"],
        ) = self._get_component_data_and_units(
            electrons["weighting"], io.Record_Component.SCALAR
        )

        return data, units, dimensions

    @staticmethod
    def _get_component_data_and_units(
        data: io.ParticleSpecies, component: str
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Get data chunk, unit and unit dimension for the given component.

        :param data: particle species data
        :param component: the component ("x", "y", "z")
        :return: data chunk, unit and unit dimension
        """
        component_data = data[component]
        return component_data.load_chunk(), component_data.unit_SI, data.unit_dimension

    def _check_dimensions(self, dimensions: dict):
        for key, dimension in dimensions.items():
            attribute = key.split("_")[0]
            self._assert_dimension_close(dimension, EXPECTED_DIMS[attribute])

    @staticmethod
    def _assert_dimension_close(actual: np.ndarray, expected: np.ndarray):
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)

    def _rescale_momenta(self, data: dict, weighting_power: float) -> dict:
        for component in COMPONENTS:
            data[f"momentum_{component}"] *= data["weights"] ** (-weighting_power)
        return data

    def _create_dataframe(self, data: dict, units: dict) -> pd.DataFrame:
        df = pd.DataFrame(data)
        # convert all columns to SI units
        for column, unit in units.items():
            df[column] *= unit
        return df

    @staticmethod
    def _update_data_frame(df: pd.DataFrame, swap_axes: bool) -> pd.DataFrame:
        df = HDF5Reader._add_offsets_to_positions(df)
        df = HDF5Reader._swap_yz_axes(df) if swap_axes else df
        df = HDF5Reader._convert_position_units(df)
        df = HDF5Reader._convert_momentum_units(df)
        df = HDF5Reader._add_energy_column(df)
        return df

    @staticmethod
    def _add_energy_column(df: pd.DataFrame) -> pd.DataFrame:
        df["energy_mev"] = np.sqrt(
            df["momentum_x_mev_c"] ** 2
            + df["momentum_y_mev_c"] ** 2
            + df["momentum_z_mev_c"] ** 2
            + electron_mass_MeV_c2**2
        )
        return df

    @staticmethod
    def _add_offsets_to_positions(df: pd.DataFrame) -> pd.DataFrame:
        for component in COMPONENTS:
            df[f"position_{component}"] += df[f"positionOffset_{component}"]
        return df.drop(
            columns=[f"positionOffset_{component}" for component in COMPONENTS]
        )

    @staticmethod
    def _convert_position_units(df: pd.DataFrame) -> pd.DataFrame:
        new_column_names = {}
        for component in COMPONENTS:
            df[f"position_{component}"] *= POSITION_CONVERSION_FACTOR
            new_column_names[f"position_{component}"] = f"position_{component}_um"
        return df.rename(columns=new_column_names)

    @staticmethod
    def _convert_momentum_units(df: pd.DataFrame) -> pd.DataFrame:
        new_column_names = {}
        for component in COMPONENTS:
            df[f"momentum_{component}"] *= MOMENTUM_CONVERSION_FACTOR
            new_column_names[f"momentum_{component}"] = f"momentum_{component}_mev_c"
        return df.rename(columns=new_column_names)

    @staticmethod
    def _swap_yz_axes(df: pd.DataFrame) -> pd.DataFrame:
        for attribute in ["position", "momentum"]:
            df[[f"{attribute}_y", f"{attribute}_z"]] = df[
                [f"{attribute}_z", f"{attribute}_y"]
            ]
        logger.info("Swapping y and z axes.\n")
        return df

    def _data_stats(self, df: pd.DataFrame):
        logger.info("The particle bunch is propagating along the z direction.\n")

        dataset_info(df)

        weighted_average_energy = np.average(df["energy_mev"], weights=df["weights"])
        logger.info(
            "The (weighted) mean energy is %.6e MeV.\n", weighted_average_energy
        )
