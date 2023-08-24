import argparse
from pathlib import Path

from opmdtogeant.df_to_txt import DataFrameToFile
from opmdtogeant.reader import HDF5Reader, electron_mass_MeV_c2
from opmdtogeant.visualize_phase_space import PhaseSpaceVisualizer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=str, help="Path to the HDF5 file")
    args = parser.parse_args()
    h5_path = Path(args.h5_path)

    # Create the dataframe
    h5_reader = HDF5Reader(h5_path, "e_highGamma")
    df = h5_reader.build_df()

    # Write the dataframe to a file
    df_to_file = DataFrameToFile(df)
    # TODO: uncomment
    # df_to_file.write_to_file(h5_path.with_suffix(".txt")) # slow

    # Create the phase space plots
    phase_space = PhaseSpaceVisualizer(
        dataframe=df,
        weight_column="weights",
        energy_column="energy_mev",
        energy_label="Energy (MeV)",
    )
    phase_space.create_combined_png()


if __name__ == "__main__":
    main()
