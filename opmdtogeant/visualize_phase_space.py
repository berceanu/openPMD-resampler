import itertools
import os
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm

from opmdtogeant.figure_layout import FigureCreator
from opmdtogeant.utils import combine_images, unique_filename


class PhaseSpaceVisualizer:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        weight_column: str = "weights",
        energy_column: str = "energy_mev",
        energy_label: Optional[str] = None,
        vmax: Optional[float] = None,
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

        if self.energy_column == "energy_mev":
            self.energy_label = "Energy (MeV)"
        elif energy_label is not None:
            self.energy_label = energy_label
        else:
            raise ValueError("Must supply energy_label.")

        if vmax is None:
            vmax = self.calculate_vmax()
        self._vmax = vmax

        self.norm = LogNorm(
            vmin=self.dataframe[self.weight_column].min(), vmax=self.vmax
        )

    def calculate_vmax(self):
        bin_edges = np.histogram_bin_edges(self.dataframe["momentum_x_mev_c"], bins=200)
        counts, _ = np.histogram(
            self.dataframe["momentum_x_mev_c"],
            bins=bin_edges,
            weights=self.dataframe["weights"],
        )
        order_of_magnitude = np.floor(np.log10(counts.max()))
        ten_to_the_order = np.power(10, order_of_magnitude)
        return ten_to_the_order

    @property
    def vmax(self):
        return self._vmax

    @vmax.setter
    def vmax(self, value):
        self._vmax = value

    def save_comparative_fig(
        self, other, output_filename: str = "comparative_phase_space.png"
    ):
        if not isinstance(other, PhaseSpaceVisualizer):
            raise TypeError("Both objects must be instances of PhaseSpaceVisualizer")

        if set(self.dataframe.columns) != set(other.dataframe.columns):
            raise ValueError("Dataframes must have exactly the same column names.")

        filenames = [
            self.plot_bunch(title=("original", "royalblue")),
            other.plot_bunch(title=("reduced", "#FF7F0E")),
            self.plot_histograms(other),
            self.plot_emittance(title=("original", "royalblue")),
            other.plot_emittance(title=("reduced", "#FF7F0E")),
        ]

        # Combine the plots
        combine_images(filenames, output_filename)

        # Delete the temporary files
        for fname in filenames:
            os.remove(fname)

    def savefig(self, output_filename: str = "phase_space.png"):
        filenames = [self.plot_bunch(), self.plot_histograms(), self.plot_emittance()]
        combine_images(filenames, output_filename)
        for fname in filenames:
            os.remove(fname)

    def plot_histograms(
        self,
        other: Optional["PhaseSpaceVisualizer"] = None,
        output_filename: Optional[str] = None,
    ):
        if other is not None and not isinstance(other, PhaseSpaceVisualizer):
            raise TypeError("Both objects must be instances of PhaseSpaceVisualizer")

        if output_filename is None:
            output_filename = unique_filename(".png")

        fig = FigureCreator(layout=[3, 3, 2])

        # first two rows, 3 columns each
        for row, (feature_set, label_set) in enumerate(
            zip(
                (self.position_features, self.momentum_features),
                (self.position_labels, self.momentum_labels),
            )
        ):
            for col, feature in enumerate(feature_set):
                use_y_label = col == 0
                fig.histogram_plot(
                    ax_position=(row, col),
                    df=self.dataframe,
                    col=feature,
                    x_label=label_set[col],
                    use_y_label=use_y_label,
                    weight_col=self.weight_column,
                )
                if other is not None:
                    ax = fig._get_ax((row, col)).twinx()
                    fig.histogram_plot(
                        ax=ax,
                        df=other.dataframe,
                        col=feature,
                        x_label=label_set[col],
                        weight_col=other.weight_column,
                    )

        # last row, 2 columns
        fig.weight_distribution_plot(
            ax_position=(2, 0),
            df=self.dataframe,
            weight_col=self.weight_column,
            label="original",
        )
        if other is not None:
            ax = fig._get_ax((2, 0)).twinx()
            fig.weight_distribution_plot(
                ax=ax,
                df=other.dataframe,
                weight_col=other.weight_column,
                label="reduced",
            )
        fig.histogram_plot(
            ax_position=(2, 1),
            df=self.dataframe,
            col=self.energy_column,
            x_label=self.energy_label,
            weight_col=self.weight_column,
        )
        if other is not None:
            ax = fig._get_ax((2, 1)).twinx()
            fig.histogram_plot(
                ax=ax,
                df=other.dataframe,
                col=other.energy_column,
                x_label=other.energy_label,
                weight_col=other.weight_column,
            )
        fig.save_figure(output_filename)
        return output_filename

    def plot_bunch(
        self,
        output_filename: Optional[str] = None,
        title: Optional[Tuple[str, str]] = None,
    ):
        if output_filename is None:
            output_filename = unique_filename(".png")

        layout = [3, 3]
        # Assume both rows have the same number of columns
        num_cols = layout[0]
        fig = FigureCreator(layout=layout, title=title)

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
        fig.create_colorbar(self.norm)
        fig.save_figure(output_filename)
        return output_filename

    def plot_emittance(
        self,
        output_filename: Optional[str] = None,
        title: Optional[Tuple[str, str]] = None,
    ):
        if output_filename is None:
            output_filename = unique_filename(".png")

        fig = FigureCreator(layout=[3, 3, 3], title=title)

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
        fig.create_colorbar(self.norm)
        fig.save_figure(output_filename)
        return output_filename


#     def savefig(self, output_filename: str = "comparative_phase_space.png"):
#         # plot the electron bunch from the first dataframe
#         # plot the electron bunch from the second dataframe

#         # plot the combined histograms, ie both dataframes on the same plot

#         # plot the emittance from the first dataframe
#         # plot the emittance from the second dataframe

#         # combine all the PNGs into the final PNG
#         # self.combine_pngs(output_filename)
