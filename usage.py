import argparse
from pathlib import Path

from openpmd_viewer.addons import LpaDiagnostics

from opmdtogeant.reader import HDF5Reader, electron_mass_MeV_c2
from opmdtogeant.df_to_txt import DataFrameToFile


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=str, help="Path to the HDF5 file")
    args = parser.parse_args()

    # Create the dataframe
    h5_path = Path(args.h5_path)
    h5_reader = HDF5Reader(str(h5_path))
    df = h5_reader.read_file()

    # Write the dataframe to a file
    df_to_file = DataFrameToFile(df)
    df_to_file.write_to_file(h5_path.with_suffix(".txt"))

    # Sanity check
    diags = LpaDiagnostics(str(h5_path), check_all_files=False)
    # a = diags.get_energy_spread(iteration=126175, species="e_all", property="energy")
    mean_gamma = diags.get_mean_gamma(iteration=126175, species="e_all")[0]
    mean_gamma_mev = mean_gamma * electron_mass_MeV_c2
    print(f"\nopenPMD-viewer's (weighted) mean energy is {mean_gamma_mev:.6e} MeV.\n")


if __name__ == "__main__":
    main()
