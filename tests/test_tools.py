"""Tests for the command-line tools next to the package: read_fortran.py and compare_calorimeter.py."""
import numpy as np
import pytest

from compare_calorimeter import compare
from openpmd_resampler.calorimeter import CalorimeterSampler
from openpmd_resampler.df_to_txt import DataFrameToFile
from read_fortran import read_fortran_particles
from test_calorimeter import PROTON_MASS, synthetic_calorimeter


@pytest.fixture
def particles():
    return CalorimeterSampler(synthetic_calorimeter(), None, PROTON_MASS).sample(
        mode="per_bin", particles_per_bin=2
    )


def test_fortran_default_layout_reads_weights(particles, tmp_path):
    path = tmp_path / "default.dat"
    DataFrameToFile(particles).exclude_energy().momentum_in_mc(PROTON_MASS).write_to_file(
        path, fortran_unformatted=True
    )
    df = read_fortran_particles(path)
    assert list(df.columns) == ["x_m", "y_m", "z_m", "ux", "uy", "uz", "weights"]
    np.testing.assert_array_equal(df["weights"], particles["weights"])


def test_fortran_layout_without_weights_can_be_named(particles, tmp_path):
    """exclude_weights() with the energy kept also gives 7 records."""
    path = tmp_path / "energy.dat"
    DataFrameToFile(particles).exclude_weights().momentum_in_mc(PROTON_MASS).write_to_file(
        path, fortran_unformatted=True
    )
    columns = ["x_m", "y_m", "z_m", "ux", "uy", "uz", "kinetic_energy_mev"]
    df = read_fortran_particles(path, columns=columns)
    assert list(df.columns) == columns
    np.testing.assert_array_equal(df["kinetic_energy_mev"], particles["kinetic_energy_mev"])

    with pytest.raises(ValueError, match="7 data records, but 6 column names"):
        read_fortran_particles(path, columns=columns[:-1])


def test_compare_reports_zero_overlap_and_still_plots(particles, tmp_path, capsys):
    calorimeter = synthetic_calorimeter()
    inside = tmp_path / "inside.dat"
    DataFrameToFile(particles).write_to_file(inside)
    # Push every particle far above the calorimeter's maximum energy.
    above = particles.copy()
    for column in ["momentum_x_mev_c", "momentum_y_mev_c", "momentum_z_mev_c"]:
        above[column] *= 1e4
    outside = tmp_path / "above.dat"
    DataFrameToFile(above).write_to_file(outside)

    out = tmp_path / "figures"
    compare(calorimeter, [("inside", inside), ("above", outside)], PROTON_MASS, out=out)

    printed = capsys.readouterr().out
    assert "above: zero overlap" in printed
    assert "inside: max energy" in printed
    for name in ["compare_dNdE.png", "compare_energy_yaw.png", "compare_energy_yaw_polar.png"]:
        assert (out / name).stat().st_size > 0
