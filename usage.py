import argparse
from pathlib import Path

from openpmd_viewer.addons import LpaDiagnostics

from opmdtogeant.df_to_txt import DataFrameToFile

from opmdtogeant.visualize_phase_space import PhaseSpaceVisualizer
from opmdtogeant.reader import HDF5Reader, electron_mass_MeV_c2


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=str, help="Path to the HDF5 file")
    args = parser.parse_args()
    h5_path = Path(args.h5_path)

    # Create the dataframe
    h5_reader = HDF5Reader(h5_path)
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

    # Sanity check
    # TODO: remove
    diags = LpaDiagnostics(str(h5_path), check_all_files=False)
    mean_gamma = diags.get_mean_gamma(iteration=126175, species="e_all")[0]
    mean_gamma_mev = mean_gamma * electron_mass_MeV_c2
    print(f"\nopenPMD-viewer's (weighted) mean energy is {mean_gamma_mev:.6e} MeV.\n")


if __name__ == "__main__":
    main()
