from enum import Enum
from typing import Tuple

import numpy as np
import openpmd_api as io
import pandas as pd

from opmdresampler.log import logger
from opmdresampler.units import constants, conversion_factors, expected_dims
from opmdresampler.utils import dataset_info


class Components(Enum):
    X = "x"
    Y = "y"
    Z = "z"


class Attributes(Enum):
    POSITION = "position"
    POSITION_OFFSET = "positionOffset"
    MOMENTUM = "momentum"


class OpenPMDLoader:
    def __init__(self, file_path: str, particle_species_name: str = "e_all"):
        self.file_path = str(file_path)
        self.particle_species_name = particle_species_name

        self.series = self.open_series()
        logger.info(
            "Data generated using %s %s.\n",
            self.series.software,
            self.series.software_version,
        )
        self.swap_yz = self.series.software == "PIConGPU"
        self.iteration = self.get_iteration()
        self.electrons = self.iteration.particles[self.particle_species_name]

        self.data, self.units = self.get_particle_data_and_units()

        self.series.flush()

        self.rescale_momenta()

        del self.series

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
                data[f"{attribute.value}_{component.value}"] = self.electrons[
                    attribute.value
                ][component.value].load_chunk()
                units[f"{attribute.value}_{component.value}"] = self.electrons[
                    attribute.value
                ][component.value].unit_SI
                dimensions[f"{attribute.value}_{component.value}"] = self.electrons[
                    attribute.value
                ].unit_dimension
                np.testing.assert_allclose(
                    dimensions[f"{attribute.value}_{component.value}"],
                    getattr(expected_dims, attribute.value),
                    atol=1e-9,
                    rtol=0,
                )

        data["weights"] = self.electrons["weighting"][
            io.Record_Component.SCALAR
        ].load_chunk()
        units["weights"] = self.electrons["weighting"][
            io.Record_Component.SCALAR
        ].unit_SI
        dimensions["weights"] = self.electrons["weighting"].unit_dimension
        np.testing.assert_allclose(
            dimensions["weights"], expected_dims.weights, atol=1e-9, rtol=0
        )

        return data, units

    def rescale_momenta(self):
        macro_weighted = self.electrons["momentum"].get_attribute("macroWeighted")
        weighting_power = self.electrons["momentum"].get_attribute("weightingPower")
        if (macro_weighted == 1) and (weighting_power != 0):
            for component in Components:
                self.data[f"momentum_{component.value}"] *= self.data["weights"] ** (
                    -weighting_power
                )


class DataFrameCreator:
    def __init__(self, data: dict, units: dict):
        self.data = data
        self.units = units
        self.df = pd.DataFrame(self.data)
        self.convert_to_SI()

    def convert_to_SI(self):
        for column_name, unit_SI in self.units.items():
            self.df[column_name] *= unit_SI


class DataFrameUpdater:
    def __init__(self, df: pd.DataFrame, swap_yz: bool):
        self.df = df
        self.add_position_offsets()
        if swap_yz:
            self.swap_yz_axes()
        self.convert_to_nuclear_units()
        self.add_energy_column()

    def add_position_offsets(self):
        for component in Components:
            self.df[f"position_{component.value}"] += self.df[
                f"positionOffset_{component.value}"
            ]
        self.df.drop(
            columns=[f"positionOffset_{component.value}" for component in Components],
            inplace=True,
        )

    def swap_yz_axes(self):
        for attribute in [Attributes.POSITION, Attributes.MOMENTUM]:
            self.df[[f"{attribute.value}_y", f"{attribute.value}_z"]] = self.df[
                [f"{attribute.value}_z", f"{attribute.value}_y"]
            ]
        logger.info("Swapping y and z axes.\n")

    def convert_to_nuclear_units(self):
        new_column_suffixes = {
            Attributes.POSITION.value: "um",
            Attributes.MOMENTUM.value: "mev_c",
        }
        new_column_names = {}
        for attribute in [Attributes.POSITION, Attributes.MOMENTUM]:
            for component in Components:
                self.df[f"{attribute.value}_{component.value}"] *= getattr(
                    conversion_factors, attribute.value
                )
                new_column_names[
                    f"{attribute.value}_{component.value}"
                ] = f"{attribute.value}_{component.value}_{new_column_suffixes[attribute.value]}"

        self.df.rename(columns=new_column_names, inplace=True)

    def add_energy_column(self):
        self.df["energy_mev"] = np.sqrt(
            self.df["momentum_x_mev_c"] ** 2
            + self.df["momentum_y_mev_c"] ** 2
            + self.df["momentum_z_mev_c"] ** 2
            + constants.electron_mass_mev_c2**2
        )


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.data_stats()

    def data_stats(self):
        logger.info("The particle bunch is propagating along the z direction.\n")

        dataset_info(self.df)

        weighted_average_energy = np.average(
            self.df["energy_mev"], weights=self.df["weights"]
        )
        logger.info(
            "The (weighted) mean energy is %.6e MeV.\n", weighted_average_energy
        )


class ParticleDataReader:
    def __init__(self, file_path: str, particle_species_name: str = "e_all"):
        self.loader = OpenPMDLoader(file_path, particle_species_name)
        self.creator = DataFrameCreator(self.loader.data, self.loader.units)
        self.updater = DataFrameUpdater(self.creator.df, swap_yz=self.loader.swap_yz)
        self.analyzer = DataAnalyzer(self.updater.df)

    @classmethod
    def from_file(
        cls, file_path: str, particle_species_name: str = "e_all"
    ) -> pd.DataFrame:
        reader = cls(file_path, particle_species_name)
        return reader.df

    @property
    def df(self):
        return self.analyzer.df
