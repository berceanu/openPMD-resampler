from enum import Enum
from typing import Tuple

import numpy as np
import openpmd_api as io
import pandas as pd

from .log import logger
from .units import constants, conversion_factors, expected_dims
from .utils import dataset_info


class Components(Enum):
    X = "x"
    Y = "y"
    Z = "z"


class Attributes(Enum):
    POSITION = "position"
    POSITION_OFFSET = "positionOffset"
    MOMENTUM = "momentum"


class OpenPMDLoader:
    def __init__(self, file_path: str, particle_species_name: str = "e_all",particle_species_mass: float = 1.0):
        self.file_path = str(file_path)
        self.particle_species_name = particle_species_name
        self.particle_species_mass = particle_species_mass

        self.series = self.open_series()

        try:
            software_info = self.series.software
            if hasattr(self.series, "software_version"):
                software_info += " version " + self.series.software_version
        except Exception:
            pass  # Handle the exception as needed
        logger.info("Data generated using %s.\n", software_info)

        self.swap_yz = self.series.software == "PIConGPU"
        self.iteration = self.get_iteration()

        detected_species = [name for name in self.iteration.particles]
        logger.info("Detected particle species: %s.\n", detected_species)
        if self.particle_species_name not in detected_species:
            raise ValueError(f"Provided species '{self.particle_species_name}' not available. Detected species: {detected_species}")
        self.particles = self.iteration.particles[self.particle_species_name]

        self.data, self.units = self.get_particle_data_and_units()

        self.series.flush()
        del self.series

        self.rescale_momenta()
        self.convert_to_SI()
        self.add_offsets()
        self.swap_yz_axes()
        self.convert_to_nuclear_units()
        self.add_energy_column()

        self.column_name_mappings = self.get_column_name_mappings()
        self.df = pd.DataFrame(self.data, dtype=np.float32)
        self.df.rename(columns=self.column_name_mappings, inplace=True)

        del self.data
        del self.units


    def open_series(self) -> io.Series:
        return io.Series(self.file_path, io.Access.read_only)

    def get_iteration(self) -> io.Iteration:
        iteration_numbers = tuple(self.series.iterations)
        if len(iteration_numbers) != 1:
            raise ValueError(f"{self.file_path} should contain only one iteration.")

        iteration_number = iteration_numbers[0]
        iteration = self.series.iterations[iteration_number]

        logger.info(
            "%s contains iteration %s, at %.2f ps.\n",
            self.file_path,
            iteration_number,
            iteration.time * iteration.time_unit_SI * 1e12,
        )
        return iteration

    def get_particle_data_and_units(self) -> Tuple[dict, dict]:
        data = {}
        units = {}
        dimensions = {}

        for attribute in Attributes:
            for component in Components:
                data[f"{attribute.value}_{component.value}"] = self.particles[
                    attribute.value
                ][component.value].load_chunk()
                units[f"{attribute.value}_{component.value}"] = self.particles[
                    attribute.value
                ][component.value].unit_SI
                dimensions[f"{attribute.value}_{component.value}"] = self.particles[
                    attribute.value
                ].unit_dimension
                np.testing.assert_allclose(
                    dimensions[f"{attribute.value}_{component.value}"],
                    getattr(expected_dims, attribute.value),
                    atol=1e-9,
                    rtol=0,
                )

        data["weights"] = self.particles["weighting"][
            io.Record_Component.SCALAR
        ].load_chunk()
        units["weights"] = self.particles["weighting"][
            io.Record_Component.SCALAR
        ].unit_SI
        dimensions["weights"] = self.particles["weighting"].unit_dimension
        np.testing.assert_allclose(
            dimensions["weights"], expected_dims.weights, atol=1e-9, rtol=0
        )

        return data, units

    def rescale_momenta(self):
        macro_weighted = self.particles[f"{Attributes.MOMENTUM.value}"].get_attribute(
            "macroWeighted"
        )
        weighting_power = self.particles[f"{Attributes.MOMENTUM.value}"].get_attribute(
            "weightingPower"
        )
        if (macro_weighted == 1) and (weighting_power != 0):
            for component in Components:
                self.data[
                    f"{Attributes.MOMENTUM.value}_{component.value}"
                ] *= self.data["weights"] ** (-weighting_power)

    def convert_to_SI(self):
        for column_name, unit_SI in self.units.items():
            self.data[column_name] = (self.data[column_name] * unit_SI)

    def add_offsets(self):
        for component in Components:
            self.data[f"{Attributes.POSITION.value}_{component.value}"] += self.data[
                f"{Attributes.POSITION_OFFSET.value}_{component.value}"
            ]
            del self.data[f"{Attributes.POSITION_OFFSET.value}_{component.value}"]

    def swap_yz_axes(self):
        if self.swap_yz:
            for attribute in [Attributes.POSITION, Attributes.MOMENTUM]:
                self.data[f"{attribute.value}_y"], self.data[f"{attribute.value}_z"] = (
                    self.data[f"{attribute.value}_z"],
                    self.data[f"{attribute.value}_y"]
                )
            logger.info("Swapping y and z axes.\n")

    def convert_to_nuclear_units(self):
        for attribute in [Attributes.POSITION, Attributes.MOMENTUM]:
            for component in Components:
                self.data[f"{attribute.value}_{component.value}"] *= getattr(
                    conversion_factors, attribute.value
                )

    def get_column_name_mappings(self):
        new_column_suffixes = {
            Attributes.POSITION.value: "um",
            Attributes.MOMENTUM.value: "mev_c",
        }
        column_name_mappings = {}
        for attribute in [Attributes.POSITION, Attributes.MOMENTUM]:
            for component in Components:
                column_name_mappings[
                    f"{attribute.value}_{component.value}"
                ] = f"{attribute.value}_{component.value}_{new_column_suffixes[attribute.value]}"

        return column_name_mappings

    def add_energy_column(self):
        momentum_x = self.data[f"{Attributes.MOMENTUM.value}_x"]
        momentum_y = self.data[f"{Attributes.MOMENTUM.value}_y"]
        momentum_z = self.data[f"{Attributes.MOMENTUM.value}_z"]

        mass = self.particle_species_mass * constants.electron_mass_mev_c2

        kinetic_energy_mev = np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2 + mass**2)
        self.data["kinetic_energy_mev"] = kinetic_energy_mev - mass


class DataFrameUpdater:
    def __init__(self, df_or_class_with_df):
        self._df_or_class_with_df = df_or_class_with_df

    @property
    def df(self):
        if isinstance(self._df_or_class_with_df, pd.DataFrame):
            return self._df_or_class_with_df
        return self._df_or_class_with_df.df

    def add_energy_column(self):
        mass = self.particle_species_mass * constants.electron_mass_mev_c2
        self.df["kinetic_energy_mev"] = np.sqrt(
            self.df["momentum_x_mev_c"] ** 2
            + self.df["momentum_y_mev_c"] ** 2
            + self.df["momentum_z_mev_c"] ** 2
            + mass**2
        ) - mass

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.data_stats()

    def data_stats(self):
        logger.info("The particle bunch is propagating along the z direction.\n")

        dataset_info(self.df)

        weighted_average_energy = np.average(
            self.df["kinetic_energy_mev"], weights=self.df["weights"]
        )
        logger.info(
            "The (weighted) mean energy is %.6e MeV.\n", weighted_average_energy
        )


class ParticleDataReader:
    def __init__(self, file_path: str, particle_species_name: str = "e_all", particle_species_mass: float = 1.0):
        self.loader = OpenPMDLoader(file_path, particle_species_name, particle_species_mass)
        self.updater = DataFrameUpdater(self.loader.df)
        self.analyzer = DataAnalyzer(self.updater.df)

    @classmethod
    def from_file(
        cls, file_path: str, particle_species_name: str = "e_all", particle_species_mass: float = 1.0
    ) -> pd.DataFrame:
        reader = cls(file_path, particle_species_name, particle_species_mass)
        return reader.df

    @property
    def df(self):
        return self.analyzer.df
