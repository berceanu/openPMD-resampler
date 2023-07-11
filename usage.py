import argparse
from pathlib import Path

from opmdtogeant.reader import HDF5Reader


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=str, help="Path to the HDF5 file")
    args = parser.parse_args()

    # Store the HDF5 file path in a variable
    h5_path = Path(args.h5_path)

    h5_reader = HDF5Reader(str(h5_path))

    df = h5_reader.read_file()


if __name__ == "__main__":
    main()
