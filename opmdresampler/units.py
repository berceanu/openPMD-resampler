"""
This module defines the units for various physical quantities used in the simulation.
"""
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import scipy.constants as const


class Units(Enum):
    POSITION = "μm"
    MOMENTUM = "MeV/c"
    WEIGHTS = "1"
    ENERGY = "MeV"


units = {
    "position_x_um": Units.POSITION,
    "position_y_um": Units.POSITION,
    "position_z_um": Units.POSITION,
    "momentum_x_mev_c": Units.MOMENTUM,
    "momentum_y_mev_c": Units.MOMENTUM,
    "momentum_z_mev_c": Units.MOMENTUM,
    "weights": Units.WEIGHTS,
    "energy_mev": Units.ENERGY,
}

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class ExpectedDims:
    position: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    positionOffset: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    momentum: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    )
    weights: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )


expected_dims = ExpectedDims()


@dataclass(frozen=True)
class Constants:
    speed_of_light: float = const.c
    electron_mass_kg: float = const.m_e
    electron_charge_picocoulombs: float = const.elementary_charge * 1e12

    meters_to_microns: float = 1e6
    ev_to_mev: float = 1e6
    joule_to_ev: float = const.electron_volt

    electron_mass_mev_c2: float = field(init=False)
    momentum_mev_c: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "electron_mass_mev_c2",
            (self.electron_mass_kg * self.speed_of_light**2)
            / (self.joule_to_ev * self.ev_to_mev),
        )
        momentum_kg_m_s = 1  # given
        object.__setattr__(
            self,
            "momentum_mev_c",
            (momentum_kg_m_s * self.speed_of_light)
            / (self.joule_to_ev * self.ev_to_mev),
        )


constants = Constants()


@dataclass(frozen=True)
class ConversionFactors:
    momentum: float = constants.momentum_mev_c
    position: float = constants.meters_to_microns


conversion_factors = ConversionFactors()
