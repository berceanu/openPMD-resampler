"""Tests for generating macroparticles from a calorimeter and a density mesh.

The calorimeter is synthetic, and the generated particles are re-binned with
the PIConGPU kernel's own formulas (ParticleCalorimeterFunctors.hpp) to check
that every particle lands back in the bin it was drawn from.
"""
import numpy as np
import pytest

from openpmd_resampler.calorimeter import (
    JOULE_PER_KEV,
    CalorimeterData,
    CalorimeterSampler,
    DensityGrid,
    calorimeter_frame,
)
from openpmd_resampler.units import constants

PROTON_MASS = 1836.152 * constants.electron_mass_mev_c2


def synthetic_calorimeter(log_scale=True, pos_yaw_deg=0.0, pos_pitch_deg=0.0, seed=5):
    rng = np.random.default_rng(seed)
    shape = (34, 12, 20)  # energy, pitch, yaw
    energy = np.where(rng.random(shape) < 0.3, rng.uniform(1e-9, 1e-7, shape), 0.0)
    energy[0] = 1e-7  # underflow and overflow must be excluded
    energy[-1] = 1e-7
    return CalorimeterData(
        energy_joule=energy,
        min_energy_kev=10.0 if log_scale else 0.0,
        max_energy_kev=1e4,
        log_scale=log_scale,
        max_yaw_deg=45.0,
        max_pitch_deg=20.0,
        pos_yaw_deg=pos_yaw_deg,
        pos_pitch_deg=pos_pitch_deg,
        iteration=100,
    )


def synthetic_density():
    values = np.zeros((6, 5, 4), dtype=np.float32)  # z, y, x
    values[1:4, 2, 1:3] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    return DensityGrid(
        values=values,
        axis_order=("z", "y", "x"),
        cell_size_m={"x": 1e-6, "y": 2e-6, "z": 3e-6},
        origin_m={"x": -5e-6, "y": 0.0, "z": 1e-5},
        iteration=100,
    )


def rebin_like_picongpu(df, calorimeter, mass):
    """Bin indices (energy, pitch, yaw) the PIConGPU kernel would assign."""
    # Undo the y/z swap to get back to simulation coordinates.
    momentum = df[["momentum_x_mev_c", "momentum_z_mev_c", "momentum_y_mev_c"]].to_numpy(np.float64)
    direction = momentum / np.linalg.norm(momentum, axis=1)[:, None]
    frame = calorimeter_frame(calorimeter.pos_yaw_deg, calorimeter.pos_pitch_deg)
    direction = direction @ frame.T
    yaw = np.arctan2(direction[:, 0], direction[:, 1])
    pitch = np.arcsin(np.clip(direction[:, 2], -1.0, 1.0))
    number_energy, number_pitch, number_yaw = calorimeter.energy_joule.shape
    max_yaw = np.deg2rad(calorimeter.max_yaw_deg)
    max_pitch = np.deg2rad(calorimeter.max_pitch_deg)
    yaw_bin = np.floor((0.5 + 0.5 * yaw / max_yaw) * number_yaw).astype(int)
    pitch_bin = np.floor((0.5 + 0.5 * pitch / max_pitch) * number_pitch).astype(int)

    p2 = np.sum(momentum**2, axis=1)
    kinetic_kev = 1e3 * (np.sqrt(p2 + mass**2) - mass)
    if calorimeter.log_scale:
        value = np.log10(kinetic_kev)
        low, high = np.log10(calorimeter.min_energy_kev), np.log10(calorimeter.max_energy_kev)
    else:
        value = kinetic_kev
        low, high = calorimeter.min_energy_kev, calorimeter.max_energy_kev
    energy_bin = np.floor((value - low) / (high - low) * (number_energy - 2)).astype(int) + 1
    return energy_bin, pitch_bin, yaw_bin


def bins_away_from_edges(df, calorimeter, mass, tolerance=1e-4):
    """Mask of particles not within float noise of a bin edge in any coordinate."""
    momentum = df[["momentum_x_mev_c", "momentum_z_mev_c", "momentum_y_mev_c"]].to_numpy(np.float64)
    direction = momentum / np.linalg.norm(momentum, axis=1)[:, None]
    frame = calorimeter_frame(calorimeter.pos_yaw_deg, calorimeter.pos_pitch_deg)
    direction = direction @ frame.T
    number_energy, number_pitch, number_yaw = calorimeter.energy_joule.shape
    yaw_fraction = (0.5 + 0.5 * np.arctan2(direction[:, 0], direction[:, 1])
                    / np.deg2rad(calorimeter.max_yaw_deg)) * number_yaw
    pitch_fraction = (0.5 + 0.5 * np.arcsin(np.clip(direction[:, 2], -1, 1))
                      / np.deg2rad(calorimeter.max_pitch_deg)) * number_pitch
    kinetic_kev = 1e3 * (np.sqrt(np.sum(momentum**2, axis=1) + mass**2) - mass)
    if calorimeter.log_scale:
        value = np.log10(kinetic_kev)
        low, high = np.log10(calorimeter.min_energy_kev), np.log10(calorimeter.max_energy_kev)
    else:
        value, low, high = kinetic_kev, calorimeter.min_energy_kev, calorimeter.max_energy_kev
    energy_fraction = (value - low) / (high - low) * (number_energy - 2)
    distance = lambda f: np.abs(f - np.round(f))
    return (
        (distance(yaw_fraction) > tolerance)
        & (distance(pitch_fraction) > tolerance)
        & (distance(energy_fraction) > tolerance)
    )


@pytest.mark.parametrize("log_scale", [True, False])
@pytest.mark.parametrize("orientation", [(0.0, 0.0), (30.0, -15.0)])
def test_per_bin_particles_land_in_their_source_bin(log_scale, orientation):
    calorimeter = synthetic_calorimeter(log_scale, *orientation)
    sampler = CalorimeterSampler(calorimeter, synthetic_density(), PROTON_MASS)
    table = sampler.bin_table()
    df = sampler.sample(mode="per_bin", particles_per_bin=3)

    energy_bin, pitch_bin, yaw_bin = rebin_like_picongpu(df, calorimeter, PROTON_MASS)
    keep = bins_away_from_edges(df, calorimeter, PROTON_MASS)
    assert keep.mean() > 0.95
    occupied = np.argwhere(calorimeter.energy_joule[1:-1] > 0) + [1, 0, 0]
    expected = np.repeat(occupied, 3, axis=0)
    np.testing.assert_array_equal(energy_bin[keep], expected[keep, 0])
    np.testing.assert_array_equal(pitch_bin[keep], expected[keep, 1])
    np.testing.assert_array_equal(yaw_bin[keep], expected[keep, 2])

    # Weight is exact per bin and in total.
    weights = df["weights"].to_numpy(np.float64)
    np.testing.assert_allclose(weights.reshape(-1, 3).sum(axis=1), table["n_real"], rtol=1e-6)


@pytest.mark.parametrize("mode", ["random", "per_bin"])
def test_weight_and_energy_are_conserved(mode):
    calorimeter = synthetic_calorimeter()
    sampler = CalorimeterSampler(calorimeter, synthetic_density(), PROTON_MASS)
    table = sampler.bin_table()
    df = sampler.sample(mode=mode, num_macroparticles=2_000_000, particles_per_bin=50)

    weights = df["weights"].to_numpy(np.float64)
    kinetic = df["kinetic_energy_mev"].to_numpy(np.float64)
    assert weights.sum() == pytest.approx(table["n_real"].sum(), rel=1e-6)
    energy_in = calorimeter.energy_joule[1:-1].sum()
    # Statistical only: the energy is carried by rare high-energy samples.
    assert (weights * kinetic).sum() * 1e3 * JOULE_PER_KEV == pytest.approx(energy_in, rel=1e-2)
    assert list(df.columns) == [
        "position_x_m", "position_y_m", "position_z_m",
        "momentum_x_mev_c", "momentum_y_mev_c", "momentum_z_mev_c",
        "weights", "kinetic_energy_mev",
    ]
    assert (df.dtypes == np.float32).all()


def test_random_mode_reproduces_bin_populations():
    calorimeter = synthetic_calorimeter()
    sampler = CalorimeterSampler(calorimeter, None, PROTON_MASS)
    table = sampler.bin_table()
    df = sampler.sample(mode="random", num_macroparticles=2_000_000)
    energy_bin, pitch_bin, yaw_bin = rebin_like_picongpu(df, calorimeter, PROTON_MASS)
    counts = np.zeros(calorimeter.energy_joule.shape)
    np.add.at(counts, (energy_bin, pitch_bin, yaw_bin), df["weights"].to_numpy(np.float64))
    expected = np.zeros_like(counts)
    occupied = np.argwhere(calorimeter.energy_joule[1:-1] > 0) + [1, 0, 0]
    expected[tuple(occupied.T)] = table["n_real"]
    # Compare the marginal energy spectrum within 5 sigma of multinomial noise.
    sampled, target = counts.sum(axis=(1, 2)), expected.sum(axis=(1, 2))
    weight = df["weights"].iloc[0]
    sigma = weight * np.sqrt(target / weight)
    assert (np.abs(sampled - target) <= 5 * sigma + 1e-9).all()


def test_forward_direction_is_plus_z():
    calorimeter = synthetic_calorimeter()
    calorimeter.energy_joule[:] = 0.0
    calorimeter.energy_joule[10, 5:7, 9:11] = 1e-8  # around yaw = pitch = 0
    df = CalorimeterSampler(calorimeter, None, PROTON_MASS).sample(num_macroparticles=1000)
    momentum = df[["momentum_x_mev_c", "momentum_y_mev_c", "momentum_z_mev_c"]].to_numpy()
    assert (momentum[:, 2] > 0.9 * np.linalg.norm(momentum, axis=1)).all()


def test_positions_fall_in_occupied_density_cells():
    density = synthetic_density()
    df = CalorimeterSampler(synthetic_calorimeter(), density, PROTON_MASS).sample(
        num_macroparticles=200_000
    )
    # Undo the y/z swap, then map back to array indices in (z, y, x) order.
    position = {
        "x": df["position_x_m"].to_numpy(np.float64),
        "y": df["position_z_m"].to_numpy(np.float64),
        "z": df["position_y_m"].to_numpy(np.float64),
    }
    index = tuple(
        np.floor((position[a] - density.origin_m[a]) / density.cell_size_m[a]).astype(int)
        for a in density.axis_order
    )
    assert (density.values[index] > 0).all()
    # Cell populations follow the density.
    counts = np.zeros(density.values.shape)
    np.add.at(counts, index, 1.0)
    occupied = density.values > 0
    np.testing.assert_allclose(
        counts[occupied] / counts.sum(),
        density.values[occupied] / density.values.sum(),
        rtol=0.05,
    )


def test_energy_window_excludes_bins():
    calorimeter = synthetic_calorimeter()
    sampler = CalorimeterSampler(calorimeter, None, PROTON_MASS, energy_range_mev=(0.1, 1.0))
    df = sampler.sample(mode="per_bin")
    kinetic = df["kinetic_energy_mev"].to_numpy(np.float64)
    assert kinetic.min() >= 0.1 * (1 - 1e-6)
    assert kinetic.max() <= 1.0 * (1 + 1e-6)


def test_photons_have_momentum_equal_to_energy():
    df = CalorimeterSampler(synthetic_calorimeter(), None, 0.0).sample(num_macroparticles=1000)
    momentum = df[["momentum_x_mev_c", "momentum_y_mev_c", "momentum_z_mev_c"]].to_numpy(np.float64)
    np.testing.assert_allclose(
        np.linalg.norm(momentum, axis=1), df["kinetic_energy_mev"], rtol=1e-5
    )
