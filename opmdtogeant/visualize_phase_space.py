import itertools

import pandas as pd
from matplotlib.colors import LogNorm
from PIL import Image

from opmdtogeant.figure_layout import FigureCreator


class PhaseSpaceVisualizer:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        weight_column: str = "weights",
        energy_column: str = "energy_mev",
        energy_label: str = "Energy (MeV)",
    ):
        self.dataframe = dataframe
        self.position_features = ("position_x_um", "position_y_um", "position_z_um")
        self.momentum_features = (
            "momentum_x_mev_c",
            "momentum_y_mev_c",
            "momentum_z_mev_c",
        )
        self.position_labels = (r"$x$ ($\mu$m)", r"$y$ ($\mu$m)", r"$z$ ($\mu$m)")
        self.momentum_labels = (
            r"$p_{\mathrm{x}}$ (MeV/c)",
            r"$p_{\mathrm{y}}$ (MeV/c)",
            r"$p_{\mathrm{z}}$ (MeV/c)",
        )
        self.weight_column = weight_column
        self.energy_column = energy_column
        self.energy_label = energy_label
        self.norm = LogNorm(vmin=self.dataframe[self.weight_column].min(), vmax=1e8)
        self.histograms_filename = None
        self.bunch_filename = None
        self.emittance_filename = None

    def create_combined_png(self, output_filename: str = "phase_space.png"):
        self.plot_bunch()
        self.plot_histograms()
        self.plot_emittance()
        self.combine_pngs(output_filename)

    def combine_pngs(self, output_filename: str = "phase_space.png"):
        filenames = [
            self.bunch_filename,
            self.histograms_filename,
            self.emittance_filename,
        ]
        images = [Image.open(filename) for filename in filenames]

        # Assume all images have the same width
        width = images[0].width
        # Height of final image
        height = sum(image.height for image in images)

        final_image = Image.new("RGB", (width, height))

        # Paste the images onto the final image vertically
        y_offset = 0
        for image in images:
            final_image.paste(image, (0, y_offset))
            y_offset += image.height

        # Save the final image as a PNG
        final_image.save(output_filename)

    def plot_histograms(self, output_filename: str = "332.png"):
        fig = FigureCreator(layout=[3, 3, 2])

        # first two rows, 3 columns each
        for row, (feature_set, label_set) in enumerate(
            zip(
                (self.position_features, self.momentum_features),
                (self.position_labels, self.momentum_labels),
            )
        ):
            for col, feature in enumerate(feature_set):
                fig.histogram_plot(
                    ax_position=(row, col),
                    df=self.dataframe,
                    col=feature,
                    x_label=label_set[col],
                    weight_col=self.weight_column,
                )

        # last row, 2 columns
        fig.weight_distribution_plot(
            ax_position=(2, 0), df=self.dataframe, weight_col=self.weight_column
        )
        fig.histogram_plot(
            ax_position=(2, 1),
            df=self.dataframe,
            col=self.energy_column,
            x_label=self.energy_label,
            weight_col=self.weight_column,
        )
        fig.save_figure(output_filename)
        self.histograms_filename = output_filename

    def plot_bunch(self, output_filename: str = "2x3.png"):
        layout = [3, 3]
        num_rows = len(layout)
        num_cols = layout[0]
        fig = FigureCreator(layout=layout)

        coords = {
            coord: (feature, label)
            for coord, feature, label in zip(
                ["x", "y", "z"], self.position_features, self.position_labels
            )
        }
        p = {
            coord: (feature, label)
            for coord, feature, label in zip(
                ["x", "y", "z"], self.momentum_features, self.momentum_labels
            )
        }
        axes_pos = (
            (coords["x"], coords["y"]),
            (coords["z"], coords["x"]),
            (coords["z"], coords["y"]),
        )
        axes_mom = ((p["x"], p["y"]), (p["x"], p["z"]), (p["y"], p["z"]))
        combined_axes = itertools.chain(axes_pos, axes_mom)

        for i, ((x_col, x_label), (y_col, y_label)) in enumerate(combined_axes):
            row = i // num_cols
            col = i % num_cols
            fig.shade_plot(
                ax_position=(row, col),
                df=self.dataframe,
                x_col=x_col,
                y_col=y_col,
                x_label=x_label,
                y_label=y_label,
                weight_col=self.weight_column,
                norm=self.norm,
            )
        fig.create_colorbar()
        fig.save_figure(output_filename)
        self.bunch_filename = output_filename

    def plot_emittance(self, output_filename: str = "3x3.png"):
        fig = FigureCreator(layout=[3, 3, 3])

        for row, (position, position_label) in enumerate(
            zip(self.position_features, self.position_labels)
        ):
            for col, (momentum, momentum_label) in enumerate(
                zip(self.momentum_features, self.momentum_labels)
            ):
                fig.shade_plot(
                    ax_position=(row, col),
                    df=self.dataframe,
                    x_col=position,
                    y_col=momentum,
                    x_label=position_label,
                    y_label=momentum_label,
                    weight_col=self.weight_column,
                    norm=self.norm,
                )
        fig.create_colorbar()
        fig.save_figure(output_filename)
        self.emittance_filename = output_filename
