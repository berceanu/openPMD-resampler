import itertools
import os
from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from opmdresampler.figure_layout import FigureLayout
from opmdresampler.histograms import EqualWeightDistributionPlot, StandardHistogramPlot
from opmdresampler.image_plots import StandardDataShaderPlot
from opmdresampler.plot_utils import PURPLE_RABBIT
from opmdresampler.utils import combine_images, unique_filename


class MultiplePanelPlotter(ABC):
    layout = None
    weight_col = "weights"

    def __init__(
        self,
        df,
        position_features,
        position_labels,
        momentum_features,
        momentum_labels,
        output_filename=None,
    ):
        self.fig_layout = None
        self.df = df
        self.position_features = position_features
        self.position_labels = position_labels
        self.momentum_features = momentum_features
        self.momentum_labels = momentum_labels

        if output_filename is not None:
            self.output_filename = output_filename
        else:
            self.output_filename = unique_filename(".png")

    @abstractmethod
    def plot_panels(self):
        """
        Abstract method to plot panels.
        This method needs to be implemented in each child class.
        One needs to loop through the position and momentum features
        and use the plotter on each panel.
        """

    def create_plot(self):
        self.fig_layout = FigureLayout(layout=self.layout)
        self.plot_panels()
        return self

    def savefig(self):
        self.fig_layout.savefig(self.output_filename)
        return self.output_filename


class MultipleHistogramPlotter(MultiplePanelPlotter):
    layout = (3, 3, 2)
    energy_col = "energy_mev"
    energy_label = "Energy (MeV)"

    def __init__(
        self,
        df,
        position_features,
        position_labels,
        momentum_features,
        momentum_labels,
        output_filename=None,
    ):
        super().__init__(
            df,
            position_features,
            position_labels,
            momentum_features,
            momentum_labels,
            output_filename,
        )
        self.all_plotters = []

    def plot_panels(self):
        self.all_plotters = []
        # first two rows, 3 columns each
        for row, (feature_set, label_set) in enumerate(
            zip(
                (self.position_features, self.momentum_features),
                (self.position_labels, self.momentum_labels),
            )
        ):
            for col, feature in enumerate(feature_set):
                ax = self.fig_layout.get_ax(row, col)
                plotter = StandardHistogramPlot(ax, self.df, feature, label_set[col])
                plotter.weight_col = self.weight_col
                plotter.create_plot()
                self.all_plotters.append(plotter)
                if col != 0:
                    ax.set_ylabel("")

        # last row, 2 columns
        ax = self.fig_layout.get_ax(2, 0)
        plotter = EqualWeightDistributionPlot(ax, self.df)
        plotter.weight_col = self.weight_col
        plotter.create_plot()
        self.all_plotters.append(plotter)

        ax = self.fig_layout.get_ax(2, 1)
        plotter = StandardHistogramPlot(ax, self.df, self.energy_col, self.energy_label)
        plotter.weight_col = self.weight_col
        plotter.create_plot()
        self.all_plotters.append(plotter)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        attributes_to_check = [
            "layout",
            "weight_col",
            "energy_col",
            "energy_label",
            "position_features",
            "position_labels",
            "momentum_features",
            "momentum_labels",
        ]

        for attr in attributes_to_check:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise ValueError(f"Can only add another {type(self)} instance.")

        if self != other:
            raise ValueError(
                f"The two {type(self)} instances do not have equal attributes."
            )

        if len(self.all_plotters) != len(other.all_plotters):
            raise ValueError(
                "The number of elements in 'self.all_plotters' and "
                "'other.all_plotters' are not equal."
            )

        for self_plotter, other_plotter in zip(self.all_plotters, other.all_plotters):
            self_plotter += other_plotter

        return self


class MultipleImagePlotter(MultiplePanelPlotter):
    cmap = PURPLE_RABBIT
    cbar_label = "Number of 'real' electrons"
    norm = None

    def compute_vmax(self):
        std_hist = StandardHistogramPlot(ax=None, df=self.df, col=self.df.columns[0])
        std_hist.weight_col = self.weight_col
        _, density = std_hist.compute_histogram()

        order_of_magnitude = np.floor(np.log10(density.max()))
        return np.power(10, order_of_magnitude)

    def compute_norm(self):
        if self.norm is None:
            vmax = self.compute_vmax()
            self.norm = LogNorm(vmin=self.df[self.weight_col].min(), vmax=vmax)
        return self.norm

    def add_colorbar(self):
        self.fig_layout.fig.subplots_adjust(right=0.91)
        cax = self.fig_layout.fig.add_axes([0.92, 0.15, 0.025, 0.7])

        # Create a "fake" ScalarMappable with the colormap and norm
        sm = ScalarMappable(cmap=self.cmap, norm=self.norm)
        sm.set_array([])  # Make the image invisible

        # Now we can create the colorbar
        cbar = self.fig_layout.fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label(self.cbar_label)
        return cbar

    def create_plot(self):
        super().create_plot()
        self.compute_norm()
        self.add_colorbar()
        return self


class EmittancePlotter(MultipleImagePlotter):
    layout = (3, 3, 3)

    def plot_panels(self):
        for row, (position, position_label) in enumerate(
            zip(self.position_features, self.position_labels)
        ):
            for col, (momentum, momentum_label) in enumerate(
                zip(self.momentum_features, self.momentum_labels)
            ):
                ax = self.fig_layout.get_ax(row, col)
                plotter = StandardDataShaderPlot(
                    ax, self.df, position, momentum, position_label, momentum_label
                )
                plotter.create_plot(add_cbar=False)


class BunchPlotter(MultipleImagePlotter):
    layout = (3, 3)

    def plot_panels(self):
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

        num_cols = self.layout[0]
        for i, ((x_col, x_label), (y_col, y_label)) in enumerate(combined_axes):
            row = i // num_cols
            col = i % num_cols
            ax = self.fig_layout.get_ax(row, col)
            plotter = StandardDataShaderPlot(
                ax, self.df, x_col, y_col, x_label, y_label
            )
            plotter.create_plot(add_cbar=False)


class PhaseSpaceVisualizer:
    def __init__(
        self,
        df: pd.DataFrame,
        weight_col: str = "weights",
        energy_col: str = "energy_mev",
        energy_label: Optional[str] = "Energy (MeV)",
    ):
        self.df = df
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
        self.weight_col = weight_col
        self.energy_col = energy_col
        self.energy_label = energy_label

        self.bunch_plotter = BunchPlotter(
            self.df,
            self.position_features,
            self.position_labels,
            self.momentum_features,
            self.momentum_labels,
        )
        self.bunch_plotter.weight_col = self.weight_col
        #
        self.emittance_plotter = EmittancePlotter(
            self.df,
            self.position_features,
            self.position_labels,
            self.momentum_features,
            self.momentum_labels,
        )
        self.emittance_plotter.weight_col = self.weight_col
        #
        self.histograms_plotter = MultipleHistogramPlotter(
            self.df,
            self.position_features,
            self.position_labels,
            self.momentum_features,
            self.momentum_labels,
        )
        self.histograms_plotter.weight_col = self.weight_col
        self.histograms_plotter.energy_col = self.energy_col
        self.histograms_plotter.energy_label = self.energy_label

        self.filenames = [
            self.bunch_plotter.create_plot().savefig(),
            self.histograms_plotter.create_plot().savefig(),
            self.emittance_plotter.create_plot().savefig(),
        ]

    def savefig(self, output_filename: str):
        # Combine the plots
        combine_images(self.filenames, output_filename)
        # Delete the temporary files
        for fname in self.filenames:
            os.remove(fname)

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise ValueError(f"Can only add another {type(self)} instance.")

        if set(self.df.columns) != set(other.df.columns):
            raise ValueError(
                f"The two {type(self)} instances do not have the same DataFrame column names."
            )

        def add_title_and_save_plotter(plotter, title):
            plot = plotter.create_plot()
            plot.fig_layout.fig.suptitle(title)
            return plot.savefig()

        def add_legend_and_save(plot, ax_label, twin_ax_label):
            last_plotter = plot.all_plotters[-1]
            ax = last_plotter.ax
            twin_ax = ax.get_shared_x_axes().get_siblings(ax)[0]
            ax_color = ax.lines[0].get_color()
            twin_ax_color = twin_ax.lines[0].get_color()
            ax_patch = mpatches.Patch(color=ax_color, label=ax_label)
            twin_ax_patch = mpatches.Patch(color=twin_ax_color, label=twin_ax_label)
            ax.legend(handles=[ax_patch, twin_ax_patch])
            return plot.savefig()

        self_hist = self.histograms_plotter.create_plot()
        other_hist = other.histograms_plotter.create_plot()
        self_hist += other_hist

        self.filenames = [
            add_title_and_save_plotter(self.bunch_plotter, "Original"),
            add_title_and_save_plotter(other.bunch_plotter, "Reduced"),
            add_legend_and_save(self_hist, "Original", "Reduced"),
            add_title_and_save_plotter(self.emittance_plotter, "Original"),
            add_title_and_save_plotter(other.emittance_plotter, "Reduced"),
        ]

        return self
