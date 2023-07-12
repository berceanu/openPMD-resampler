import argparse
from pathlib import Path

from openpmd_viewer.addons import LpaDiagnostics

from opmdtogeant.df_to_txt import DataFrameToFile
from opmdtogeant.figure_layout import FigureCreator
from opmdtogeant.particle_visualizer import ParticleVisualizer
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
    # df_to_file = DataFrameToFile(df)
    # df_to_file.write_to_file(h5_path.with_suffix(".txt"))

    # Visualize the dataframe
    features = [
        "position_x_um",
        "position_y_um",
        "position_z_um",
        "momentum_x_mev_c",
        "momentum_y_mev_c",
        "momentum_z_mev_c",
    ]
    weights_column = "weights"

    # visualizer = ParticleVisualizer(df, features, weights_column)
    # visualizer.visualize("output.png")

    # Sanity check
    diags = LpaDiagnostics(str(h5_path), check_all_files=False)
    # a = diags.get_energy_spread(iteration=126175, species="e_all", property="energy")
    mean_gamma = diags.get_mean_gamma(iteration=126175, species="e_all")[0]
    mean_gamma_mev = mean_gamma * electron_mass_MeV_c2
    print(f"\nopenPMD-viewer's (weighted) mean energy is {mean_gamma_mev:.6e} MeV.\n")

    # Screen resolution in pixels
    W_px = 1440
    H_px = 900

    # Screen size in inches
    D_in = 13.3

    # Compute DPI
    dpi = FigureCreator.compute_dpi(W_px, H_px, D_in)

    # Define layout
    layout = [3, 3, 2]

    # Create a FigureCreator instance
    fig_creator = FigureCreator(W_px, H_px, dpi, layout)

    # Create figure and subplots
    fig, axs = fig_creator.create_figure_and_subplots()

    # Apply x and y labels to each subplot
    x_labels = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
    y_labels = ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8"]

    label_idx = 0
    for row in axs:
        for ax in row:
            ax.set_xlabel(x_labels[label_idx])
            ax.set_ylabel(y_labels[label_idx])
            label_idx += 1

    # Save the figure
    fig.savefig("3x3x2.png", dpi=dpi)


if __name__ == "__main__":
    main()
