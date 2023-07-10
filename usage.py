import argparse
from pathlib import Path

from opmdtogeant.reader import HDF5Reader
from openpmd_viewer import OpenPMDTimeSeries

MeV_c_factor = 1.8711573470618968e21


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=str, help="Path to the HDF5 file")
    args = parser.parse_args()

    # Store the HDF5 file path in a variable
    h5_path = Path(args.h5_path)

    ts = OpenPMDTimeSeries(str(h5_path))
    z, uz = ts.get_particle(
        var_list=["z", "uz"], iteration=126175, species="e_all", plot=False
    )
    print(z)
    print(uz)

    h5_reader = HDF5Reader(str(h5_path))

    df = h5_reader.read_file()
    print(df)


if __name__ == "__main__":
    main()
