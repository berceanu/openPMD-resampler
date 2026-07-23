"""Conservation tests for the resampling algorithms.

Each algorithm's primary value is conserving specific quantities:

- global leveling thinning: total weight (charge), statistically;
- Vranic merging: total weight, momentum and energy, exactly per packet;
- Voronoi merging: total weight and momentum exactly, energy up to the
  momentum spread threshold.

All totals are accumulated in float64 to keep the float32 storage of the
DataFrame from polluting the comparison.
"""
import numpy as np
import pandas as pd
import pytest

from openpmd_resampler.resampling import ParticleResampler
from openpmd_resampler.units import constants

ELECTRON_MASS = constants.electron_mass_mev_c2


def devices():
    result = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            result.append("cuda")
    except ImportError:
        pass
    return result


def synthetic_beam(n=200_000, seed=1234):
    """A relativistic electron bunch with non-uniform weights.

    Mean momentum is nonzero along every axis so that each total-momentum
    component is a meaningful conservation check (no cancellation to ~0).
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "position_x_m": rng.normal(0.0, 2e-6, n),
            "position_y_m": rng.normal(0.0, 2e-6, n),
            "position_z_m": rng.normal(0.0, 5e-6, n),
            "momentum_x_mev_c": rng.normal(3.0, 1.0, n),
            "momentum_y_mev_c": rng.normal(-2.0, 1.0, n),
            "momentum_z_mev_c": rng.normal(50.0, 5.0, n),
            "weights": rng.uniform(0.5, 2.0, n),
        },
        dtype=np.float32,
    )
    p_squared = (
        df["momentum_x_mev_c"].astype(np.float64) ** 2
        + df["momentum_y_mev_c"].astype(np.float64) ** 2
        + df["momentum_z_mev_c"].astype(np.float64) ** 2
    )
    df["kinetic_energy_mev"] = (
        np.sqrt(p_squared + ELECTRON_MASS**2) - ELECTRON_MASS
    ).astype(np.float32)
    return df


def totals(df):
    w = df["weights"].to_numpy(dtype=np.float64)
    return {
        "weight": w.sum(),
        "momentum": np.array(
            [
                (w * df[f"momentum_{c}_mev_c"].to_numpy(dtype=np.float64)).sum()
                for c in "xyz"
            ]
        ),
        "energy": (w * df["kinetic_energy_mev"].to_numpy(dtype=np.float64)).sum(),
    }


@pytest.fixture(scope="module")
def beam():
    return synthetic_beam()


@pytest.mark.parametrize("device", devices())
def test_vranic_conserves_weight_momentum_and_energy(beam, device):
    before = totals(beam)
    df_merged = (
        ParticleResampler(beam, particle_species_mass=1.0)
        .vranic_merging(
            spatial_bins=(4, 4, 4), momentum_bins=(8, 8, 8), device=device
        )
        .finalize()
    )
    after = totals(df_merged)

    assert len(df_merged) < len(beam)
    assert after["weight"] == pytest.approx(before["weight"], rel=1e-5)
    # Compare momentum against the total momentum scale, not per component,
    # so a component close to zero cannot inflate the relative error.
    momentum_scale = np.linalg.norm(before["momentum"])
    assert np.abs(after["momentum"] - before["momentum"]).max() < 1e-5 * momentum_scale
    assert after["energy"] == pytest.approx(before["energy"], rel=1e-5)
    assert list(df_merged.columns) == list(beam.columns)
    assert (df_merged.dtypes == np.float32).all()


@pytest.mark.parametrize("device", devices())
def test_voronoi_conserves_weight_and_momentum(beam, device):
    rel_mom_spread_threshold = 0.1
    before = totals(beam)
    df_merged = (
        ParticleResampler(beam, particle_species_mass=1.0)
        .voronoi_merging(
            spatial_bins=(4, 4, 4),
            min_particles_to_merge=8,
            pos_spread_threshold=0.5,
            rel_mom_spread_threshold=rel_mom_spread_threshold,
            min_mean_energy_kev=0.0,
            device=device,
        )
        .finalize()
    )
    after = totals(df_merged)

    assert len(df_merged) < len(beam)
    assert after["weight"] == pytest.approx(before["weight"], rel=1e-5)
    momentum_scale = np.linalg.norm(before["momentum"])
    assert np.abs(after["momentum"] - before["momentum"]).max() < 1e-5 * momentum_scale
    # Energy is only conserved up to the momentum spread threshold: merging a
    # cluster to its mean momentum loses the spread's contribution, of order
    # (relative spread)^2 per cluster.
    assert after["energy"] == pytest.approx(
        before["energy"], rel=rel_mom_spread_threshold**2
    )
    assert list(df_merged.columns) == list(beam.columns)
    assert (df_merged.dtypes == np.float32).all()


def test_global_leveling_thinning_conserves_weight_statistically(beam):
    k = 10.0
    before = totals(beam)
    df_thin = (
        ParticleResampler(beam, particle_species_mass=1.0)
        .global_leveling_thinning(k=k)
        .finalize()
    )
    after = totals(df_thin)

    # Survivors are kept with probability w / (k * <w>) and re-leveled to
    # k * <w>, so charge is conserved in expectation and the count drops by
    # roughly k; both hold only statistically.
    assert len(df_thin) == pytest.approx(len(beam) / k, rel=0.1)
    assert after["weight"] == pytest.approx(before["weight"], rel=0.02)
    # The thinned mean energy should stay representative of the original beam.
    mean_energy_before = before["energy"] / before["weight"]
    mean_energy_after = after["energy"] / after["weight"]
    assert mean_energy_after == pytest.approx(mean_energy_before, rel=0.02)


@pytest.mark.parametrize("device", devices())
def test_kernel_upsampling_conserves_weight_and_momentum(beam, device):
    n = 8
    before = totals(beam)
    df_up = (
        ParticleResampler(beam, particle_species_mass=1.0)
        .kernel_upsampling(
            upsampling_factor=n,
            spatial_bins=(4, 4, 4),
            position_bandwidth=0.1,
            momentum_bandwidth=0.1,
            device=device,
        )
        .finalize()
    )
    after = totals(df_up)

    # Each parent becomes exactly n daughters of weight w/n.
    assert len(df_up) == n * len(beam)
    assert after["weight"] == pytest.approx(before["weight"], rel=1e-6)
    # Antithetic pairs make the per-parent momentum sum exact up to float32
    # rounding; compare each component against the total momentum scale.
    momentum_scale = np.linalg.norm(before["momentum"])
    assert np.abs(after["momentum"] - before["momentum"]).max() < 1e-5 * momentum_scale
    # Energy is conserved only to second order in the bandwidth; the error is
    # positive by convexity of E(p) and small at bandwidth 0.1.
    assert after["energy"] == pytest.approx(before["energy"], rel=1e-3)
    assert list(df_up.columns) == list(beam.columns)
    assert (df_up.dtypes == np.float32).all()


@pytest.mark.parametrize("device", devices())
def test_kernel_upsampling_energy_error_shrinks_with_bandwidth(beam, device):
    before = totals(beam)

    def energy_error(momentum_bandwidth):
        df_up = (
            ParticleResampler(beam, particle_species_mass=1.0)
            .kernel_upsampling(
                upsampling_factor=8,
                spatial_bins=(4, 4, 4),
                position_bandwidth=0.1,
                momentum_bandwidth=momentum_bandwidth,
                device=device,
            )
            .finalize()
        )
        return abs(totals(df_up)["energy"] - before["energy"]) / before["energy"]

    # The energy error is second order in the momentum bandwidth, so a tenfold
    # smaller bandwidth gives a strictly smaller error.
    assert energy_error(0.01) < energy_error(0.1)


def offset_beam(n=200_000, seed=99):
    """A bunch with coordinates far offset from zero relative to their spread.

    Regression fixture for the weighted-variance catastrophic-cancellation
    bug found in PR #3 review: computing Var = E[x^2] - E[x]^2 in float32
    silently collapses to (near) zero once the mean dominates the spread.
    `position_z_m` (1 cm offset, 1 um spread) and `momentum_z_mev_c` (2 GeV/c
    offset, spread 10) both have offset:spread ratios the original
    synthetic `beam` fixture never exercises.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "position_x_m": rng.normal(0.0, 2e-6, n),
            "position_y_m": rng.normal(0.0, 2e-6, n),
            "position_z_m": rng.normal(1e-2, 1e-6, n),
            "momentum_x_mev_c": rng.normal(3.0, 1.0, n),
            "momentum_y_mev_c": rng.normal(-2.0, 1.0, n),
            "momentum_z_mev_c": rng.normal(2000.0, 10.0, n),
            "weights": rng.uniform(0.5, 2.0, n),
        },
        dtype=np.float32,
    )
    p_squared = (
        df["momentum_x_mev_c"].astype(np.float64) ** 2
        + df["momentum_y_mev_c"].astype(np.float64) ** 2
        + df["momentum_z_mev_c"].astype(np.float64) ** 2
    )
    df["kinetic_energy_mev"] = (
        np.sqrt(p_squared + ELECTRON_MASS**2) - ELECTRON_MASS
    ).astype(np.float32)
    return df


@pytest.mark.parametrize("device", devices())
def test_kernel_upsampling_adds_noise_to_offset_coordinates(device):
    beam = offset_beam()
    bandwidth = 0.1
    df_up = (
        ParticleResampler(beam, particle_species_mass=1.0)
        .kernel_upsampling(
            upsampling_factor=8,
            spatial_bins=(1, 1, 1),  # single cell: local spread == global spread
            position_bandwidth=bandwidth,
            momentum_bandwidth=bandwidth,
            device=device,
        )
        .finalize()
    )

    for column in ("position_z_m", "momentum_z_mev_c"):
        in_std = beam[column].to_numpy(dtype=np.float64).std()
        out_std = df_up[column].to_numpy(dtype=np.float64).std()
        added = np.sqrt(max(out_std**2 - in_std**2, 0.0))
        # Loose tolerance: this checks the kernel adds noise on the expected
        # scale (bandwidth * std), not that it collapses to ~0 as it did
        # under the float32 E[x^2] - E[x]^2 cancellation bug.
        assert added == pytest.approx(bandwidth * in_std, rel=0.5)


@pytest.mark.parametrize("device", devices())
def test_kernel_upsampling_zero_bandwidth_reproduces_parents(beam, device):
    before = totals(beam)
    df_up = (
        ParticleResampler(beam, particle_species_mass=1.0)
        .kernel_upsampling(
            upsampling_factor=8,
            spatial_bins=(4, 4, 4),
            position_bandwidth=0.0,
            momentum_bandwidth=0.0,
            device=device,
        )
        .finalize()
    )
    after = totals(df_up)

    # With no perturbation the daughters coincide with their parents, so every
    # total is conserved to float32 tolerance, energy included.
    assert len(df_up) == 8 * len(beam)
    assert after["weight"] == pytest.approx(before["weight"], rel=1e-6)
    assert after["energy"] == pytest.approx(before["energy"], rel=1e-6)


@pytest.mark.parametrize("device", devices())
def test_vranic_merging_is_reproducible_on_cpu(beam, device):
    if device != "cpu":
        pytest.skip("GPU float atomics are not bit-reproducible by design")
    kwargs = dict(spatial_bins=(4, 4, 4), momentum_bins=(8, 8, 8), device="cpu")
    first = (
        ParticleResampler(beam, particle_species_mass=1.0)
        .vranic_merging(**kwargs)
        .finalize()
    )
    second = (
        ParticleResampler(beam, particle_species_mass=1.0)
        .vranic_merging(**kwargs)
        .finalize()
    )
    pd.testing.assert_frame_equal(first, second)
