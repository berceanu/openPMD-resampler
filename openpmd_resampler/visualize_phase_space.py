import itertools
import os
from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from .figure_layout import FigureLayout
from .histograms import (
    EqualWeightDistributionPlot,
    StandardHistogramPlot,
    LogHistogramPlot,
    StandardHistogram,
    EqualWeightHistogram,
)
from .image_plots import StandardDataShaderPlot
from .plot_utils import galactic_spectrum
from .utils import combine_images, unique_filename


class MultiplePanelPlotter(ABC):
    layout = None
    weight_col = "weights"
    position_features = ("position_x_um", "position_y_um", "position_z_um")
    position_labels = (r"$x$ ($\mu$m)", r"$y$ ($\mu$m)", r"$z$ ($\mu$m)")
    momentum_features = ("momentum_x_mev_c", "momentum_y_mev_c", "momentum_z_mev_c")
    momentum_labels = (
        r"$p_{\mathrm{x}}$ (MeV/c)",
        r"$p_{\mathrm{y}}$ (MeV/c)",
        r"$p_{\mathrm{z}}$ (MeV/c)",
    )

    def __init__(
        self,
        df,
        output_filename=None,
    ):
        self.df = df
        self.output_filename = (
            output_filename if output_filename is not None else unique_filename(".png")
        )
        self.fig_layout = FigureLayout(layout=self.layout)

    @abstractmethod
    def plot_panels(self):
        """
        Abstract method to plot panels.
        This method needs to be implemented in each child class.
        One needs to loop through the position and momentum features
        and use the plotter on each panel.
        """

    def create_plot(self):
        if not hasattr(self, "plotters") or not self.plotters:
            self.plot_panels()
        return self

    def savefig(self):
        self.fig_layout.savefig(self.output_filename)
        return self.output_filename


class MultipleHistogramPlotter(MultiplePanelPlotter):
    layout = (3, 3, 2)
    energy_col = "kinetic_energy_mev"
    energy_label = "Energy (MeV)"

    def __init__(self, df, output_filename=None):
        super().__init__(df, output_filename)
        self.plotters = []

    def add_legend(self, primary_label, secondary_label):
        last_plotter = self.plotters[-1]
        primary_patch = mpatches.Patch(
            color=last_plotter.primary_color, label=primary_label
        )
        secondary_patch = mpatches.Patch(
            color=last_plotter.secondary_color, label=secondary_label
        )
        last_plotter.ax.legend(handles=[primary_patch, secondary_patch])
        return self

    def plot_panels(self):
        self.plotters = []
        # first two rows, 3 columns each
        for row, (feature_set, label_set) in enumerate(
            zip(
                (self.position_features, self.momentum_features),
                (self.position_labels, self.momentum_labels),
            )
        ):
            for col, feature in enumerate(feature_set):
                y_label = None if col == 0 else ""
                ax = self.fig_layout.get_ax(row, col)
                histogram = StandardHistogram(self.df, feature, self.weight_col)
                plotter = StandardHistogramPlot(histogram, ax, label_set[col], y_label)
                self.plotters.append(plotter)

        # last row, 2 columns
        ax = self.fig_layout.get_ax(2, 0)
        histogram = EqualWeightHistogram(self.df, self.weight_col)
        plotter = EqualWeightDistributionPlot(histogram, ax)
        self.plotters.append(plotter)

        ax = self.fig_layout.get_ax(2, 1)
        histogram = StandardHistogram(self.df, self.energy_col, self.weight_col)
        plotter = LogHistogramPlot(histogram, ax, self.energy_label)
        self.plotters.append(plotter)

    def create_plot(self):
        super().create_plot()

        for plotter in self.plotters:
            plotter.create_plot()

        return self

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

        # Call plot_panels if plotters are not populated
        if not hasattr(self, "plotters") or not self.plotters:
            self.plot_panels()
        if not hasattr(other, "plotters") or not other.plotters:
            other.plot_panels()

        if len(self.plotters) != len(other.plotters):
            raise ValueError(
                "The number of elements in 'self.plotters' and "
                "'other.plotters' are not equal."
            )

        for self_plotter, other_plotter in zip(self.plotters, other.plotters):
            self_plotter.histogram += other_plotter.histogram

        return self


class MultipleImagePlotter(MultiplePanelPlotter):
    cmap = galactic_spectrum
    cbar_label = "Number of 'real' electrons"
    norm = None

    def add_title(self, title):
        self.fig_layout.fig.suptitle(title)
        return self

    def compute_vmax(self):
        std_hist = StandardHistogram(self.df, self.df.columns[0], self.weight_col)
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
        label: Optional[str] = None,
    ):
        self.df = df
        self.label = label

        self.bunch_plotter = BunchPlotter(self.df)
        self.histogram_plotter = MultipleHistogramPlotter(self.df)
        self.emittance_plotter = EmittancePlotter(self.df)

        self.plotters = [
            self.bunch_plotter,
            self.histogram_plotter,
            self.emittance_plotter,
        ]

    def create_plot(self):
        for plotter in self.plotters:
            plotter.create_plot()
        return self

    def savefig(self, output_filename: str):
        output_filenames = []
        for plotter in self.plotters:
            fname = plotter.savefig()
            output_filenames.append(fname)

        # Combine the plots
        combine_images(output_filenames, output_filename)
        # Delete the temporary files
        for fname in output_filenames:
            os.remove(fname)

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise ValueError(f"Can only add another {type(self)} instance.")

        if set(self.df.columns) != set(other.df.columns):
            raise ValueError(
                f"The two {type(self)} instances do not have the same DataFrame column names."
            )

        comparative_histogram_plotter = self.histogram_plotter + other.histogram_plotter

        self.plotters = [
            self.bunch_plotter.add_title(self.label),
            other.bunch_plotter.add_title(other.label),
            comparative_histogram_plotter.add_legend(self.label, other.label),
            self.emittance_plotter.add_title(self.label),
            other.emittance_plotter.add_title(other.label),
        ]

        return self
