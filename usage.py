import argparse
from pathlib import Path
import pandas as pd

from opmdtogeant.reader import HDF5Reader
from openpmd_viewer import OpenPMDTimeSeries
import numpy as np

MeV_c_factor = 1.8711573470618968e21


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=str, help="Path to the HDF5 file")
    args = parser.parse_args()

    # Store the HDF5 file path in a variable
    h5_path = Path(args.h5_path)

    ts = OpenPMDTimeSeries(str(h5_path))
    x, y, z, ux, uy, uz, w = ts.get_particle(
        var_list=["x", "z", "y", "ux", "uz", "uy", "w"],
        iteration=126175,
        species="e_all",
        plot=False,
    )

    h5_reader = HDF5Reader(str(h5_path))

    df = h5_reader.read_file()

    np.testing.assert_allclose(df["position_x"], x, atol=1e-9, rtol=0)
    np.testing.assert_allclose(df["position_y"], y, atol=1e-9, rtol=0)
    np.testing.assert_allclose(df["position_z"], z, atol=1e-9, rtol=0)
    np.testing.assert_allclose(df["momentum_x"], ux, atol=0.1, rtol=1e-6)
    np.testing.assert_allclose(df["momentum_y"], uy, atol=0.1, rtol=1e-6)
    np.testing.assert_allclose(df["momentum_z"], uz, atol=0.1, rtol=1e-6)
    np.testing.assert_allclose(df["weights"], w, atol=1e-9, rtol=0)


if __name__ == "__main__":
    main()
