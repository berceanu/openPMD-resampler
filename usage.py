import argparse
from pathlib import Path

from matplotlib.colors import LogNorm
from openpmd_viewer.addons import LpaDiagnostics

from opmdtogeant.df_to_txt import DataFrameToFile
from opmdtogeant.figure_layout import FigureCreator

# from opmdtogeant.phase_space_diagrams import ParticleVisualizer
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

    selected_columns = ["position_x_um", "position_y_um", "weights"]
    new_df = df[selected_columns].copy()
    new_df.to_pickle("reduced_dataframe.pkl")

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

    # Create a FigureCreator instance
    fig_creator = FigureCreator(layout=[3, 3])

    # Calculate the global maximum and minimum
    global_min = df["weights"].min()

    # Create a LogNorm instance with the global max and min
    norm = LogNorm(vmin=global_min, vmax=1e8)

    # Top row plots
    fig_creator.shade_plot(
        ax_position=(0, 0),
        df=df,
        x_col="position_x_um",
        y_col="position_y_um",
        x_label=r"$x$ ($\mu$m)",
        y_label=r"$y$ ($\mu$m)",
        norm=norm,
    )

    fig_creator.shade_plot(
        ax_position=(0, 1),
        df=df,
        x_col="position_z_um",
        y_col="position_x_um",
        x_label=r"$z$ ($\mu$m)",
        y_label=r"$x$ ($\mu$m)",
        norm=norm,
    )

    fig_creator.shade_plot(
        ax_position=(0, 2),
        df=df,
        x_col="position_z_um",
        y_col="position_y_um",
        x_label=r"$z$ ($\mu$m)",
        y_label=r"$y$ ($\mu$m)",
        norm=norm,
    )

    # Bottom row plots
    fig_creator.shade_plot(
        ax_position=(1, 0),
        df=df,
        x_col="momentum_x_mev_c",
        y_col="momentum_y_mev_c",
        x_label=r"$p_{\mathrm{x}}$ (MeV/c)",
        y_label=r"$p_{\mathrm{y}}$ (MeV/c)",
        norm=norm,
    )

    fig_creator.shade_plot(
        ax_position=(1, 1),
        df=df,
        x_col="momentum_x_mev_c",
        y_col="momentum_z_mev_c",
        x_label=r"$p_{\mathrm{x}}$ (MeV/c)",
        y_label=r"$p_{\mathrm{z}}$ (MeV/c)",
        norm=norm,
    )

    fig_creator.shade_plot(
        ax_position=(1, 2),
        df=df,
        x_col="momentum_y_mev_c",
        y_col="momentum_z_mev_c",
        x_label=r"$p_{\mathrm{y}}$ (MeV/c)",
        y_label=r"$p_{\mathrm{z}}$ (MeV/c)",
        norm=norm,
    )

    # Create a colorbar
    fig_creator.create_colorbar()

    # Save the figure
    fig_creator.save_figure("2x3.png")

    # Create a FigureCreator instance
    fig_emittance = FigureCreator(layout=[3, 3, 3])

    fig_emittance.shade_plot(
        ax_position=(0, 0),
        df=df,
        x_col="position_x_um",
        y_col="momentum_x_mev_c",
        x_label=r"$x$ ($\mu$m)",
        y_label=r"$p_{\mathrm{x}}$ (MeV/c)",
        norm=norm,
    )
    fig_emittance.shade_plot(
        ax_position=(0, 1),
        df=df,
        x_col="position_x_um",
        y_col="momentum_y_mev_c",
        x_label=r"$x$ ($\mu$m)",
        y_label=r"$p_{\mathrm{y}}$ (MeV/c)",
        norm=norm,
    )
    fig_emittance.shade_plot(
        ax_position=(0, 2),
        df=df,
        x_col="position_x_um",
        y_col="momentum_z_mev_c",
        x_label=r"$x$ ($\mu$m)",
        y_label=r"$p_{\mathrm{z}}$ (MeV/c)",
        norm=norm,
    )
    # middle row
    fig_emittance.shade_plot(
        ax_position=(1, 0),
        df=df,
        x_col="position_y_um",
        y_col="momentum_x_mev_c",
        x_label=r"$y$ ($\mu$m)",
        y_label=r"$p_{\mathrm{x}}$ (MeV/c)",
        norm=norm,
    )
    fig_emittance.shade_plot(
        ax_position=(1, 1),
        df=df,
        x_col="position_y_um",
        y_col="momentum_y_mev_c",
        x_label=r"$y$ ($\mu$m)",
        y_label=r"$p_{\mathrm{y}}$ (MeV/c)",
        norm=norm,
    )
    fig_emittance.shade_plot(
        ax_position=(1, 2),
        df=df,
        x_col="position_y_um",
        y_col="momentum_z_mev_c",
        x_label=r"$y$ ($\mu$m)",
        y_label=r"$p_{\mathrm{z}}$ (MeV/c)",
        norm=norm,
    )
    # bottom row
    fig_emittance.shade_plot(
        ax_position=(2, 0),
        df=df,
        x_col="position_z_um",
        y_col="momentum_x_mev_c",
        x_label=r"$z$ ($\mu$m)",
        y_label=r"$p_{\mathrm{x}}$ (MeV/c)",
        norm=norm,
    )
    fig_emittance.shade_plot(
        ax_position=(2, 1),
        df=df,
        x_col="position_z_um",
        y_col="momentum_y_mev_c",
        x_label=r"$z$ ($\mu$m)",
        y_label=r"$p_{\mathrm{y}}$ (MeV/c)",
        norm=norm,
    )
    fig_emittance.shade_plot(
        ax_position=(2, 2),
        df=df,
        x_col="position_z_um",
        y_col="momentum_z_mev_c",
        x_label=r"$z$ ($\mu$m)",
        y_label=r"$p_{\mathrm{z}}$ (MeV/c)",
        norm=norm,
    )

    # Create a colorbar
    fig_emittance.create_colorbar()

    # Save the figure
    fig_emittance.save_figure("emittance.png")


if __name__ == "__main__":
    main()
