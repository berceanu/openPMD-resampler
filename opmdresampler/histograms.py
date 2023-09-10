from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np

from opmdresampler.plot_utils import add_grid, customize_tick_labels


def set_y_axis_tick_color(axes_and_colors):
    for ax, color in axes_and_colors:
        ax.tick_params(axis="y", labelcolor=color)


def set_common_x_axis_limits(ax1, ax2):
    x_min_1, x_max_1 = ax1.get_xlim()
    x_min_2, x_max_2 = ax2.get_xlim()

    x_min = min(x_min_1, x_min_2)
    x_max = max(x_max_1, x_max_2)
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)


class Histogram(ABC):
    bins = 400
    weight_col = "weights"

    def __init__(
        self,
        df,
        weight_col=None,
    ):
        self.df = df
        self.other = None
        self.weight_col = weight_col if weight_col is not None else self.weight_col

    @abstractmethod
    def compute_histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Abstract method. Must return x_coords, density.
        """

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise ValueError(f"Can only add another {type(self)} instance.")

        self.other = other
        return self


class HistogramPlot(ABC):
    primary_color = "royalblue"
    secondary_color = "#FF7F0E"  # orange
    x_label = "x"
    y_label = "y"

    def __init__(
        self,
        histogram,
        ax,
        x_label=None,
        y_label=None,
    ):
        self.histogram = histogram
        self.ax = ax
        self.x_label = x_label if x_label is not None else self.x_label
        self.y_label = y_label if y_label is not None else self.y_label

    @abstractmethod
    def plot(self, x_coords, density, color):
        """
        Abstract method. Override in child classes.
        """

    def standard_plot_styling(self):
        add_grid(self.ax)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        customize_tick_labels(self.ax)

    def twin_axis_plot_styling(self):
        self.ax.set_xlabel(None)
        self.ax.set_ylabel(None)
        customize_tick_labels(self.ax)

    def create_plot(
        self,
        plot_styling: Callable = None,
        color=None,
    ):
        if plot_styling is None:
            plot_styling = self.standard_plot_styling
        if color is None:
            color = self.primary_color
        x_coords, density = self.histogram.compute_histogram()
        self.plot(x_coords, density, color)
        plot_styling()

        if self.histogram.other is not None:
            other_plotter = self.__class__(
                self.histogram.other, self.ax.twinx(), self.x_label, self.y_label
            )
            other_plotter.create_plot(
                other_plotter.twin_axis_plot_styling, self.secondary_color
            )

            set_common_x_axis_limits(self.ax, other_plotter.ax)
            set_y_axis_tick_color(
                [
                    (self.ax, self.primary_color),
                    (other_plotter.ax, self.secondary_color),
                ]
            )

        return self

    def savefig(self, output_filename):
        self.ax.get_figure().savefig(output_filename)


class StandardHistogram(Histogram):
    def __init__(self, df, col, weight_col=None):
        super().__init__(df, weight_col)
        self.col = col

    def compute_histogram(self):
        linear_bins = np.histogram_bin_edges(self.df[self.col], bins=self.bins)

        counts, bin_edges = np.histogram(
            self.df[self.col],
            bins=linear_bins,
            weights=self.df[self.weight_col],
        )

        density = counts

        x_coords = (bin_edges[:-1] + bin_edges[1:]) / 2

        return x_coords, density


class StandardHistogramPlot(HistogramPlot):
    y_label = "Number of 'real' electrons"

    def plot(self, x_coords, density, color):
        self.ax.scatter(
            x_coords,
            density,
            s=0.5,
            c=color,
            marker="o",
            alpha=1.0,
            linewidths=0,
            edgecolors="none",
        )
        self.ax.plot(
            x_coords,
            density,
            linestyle="--",
            linewidth=0.3,
            color=color,
        )


class WeightHistogram(Histogram):
    def compute_histogram(self):
        # Define the logarithmic bins
        log_bins = np.logspace(
            np.log10(self.df[self.weight_col].min()),
            np.log10(self.df[self.weight_col].max()),
            num=self.bins,
            base=10,
        )
        counts, bin_edges = np.histogram(self.df[self.weight_col], bins=log_bins)

        # The width of the bins in log scale is the difference in log-space of the bin edges
        bin_width = np.diff(np.log10(bin_edges))

        # Now we calculate the number of entries per bin divided by the width of the bin.
        # This is equivalent to the density of entries per bin in log scale.
        density = counts / bin_width

        # Set the x coordinates of the line plot.
        # Use the midpoint of each bin as x-coordinates.
        x_coords = (bin_edges[:-1] + bin_edges[1:]) / 2

        return x_coords, density


class WeightDistributionPlot(HistogramPlot):
    x_label = "w (weights)"
    y_label = "dN/dln(w)"

    def standard_plot_styling(self):
        add_grid(self.ax)
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)

    def twin_axis_plot_styling(self):
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel(None)
        self.ax.set_ylabel(None)

    def plot(self, x_coords, density, color):
        super().plot(x_coords, density, color)
        self.ax.fill_between(x_coords, 0, density, color=color, alpha=0.3)


class EqualWeightHistogram(WeightHistogram):
    def compute_histogram(self):
        # Check if all weights are equal
        if self.df[self.weight_col].nunique() == 1:
            weight_value = self.df[self.weight_col].iloc[0]  # take the first weight

            dN = self.df.shape[0]
            dlnw = 1  # an infinitesimally small value for the natural logarithm of the same weight

            # Plot a single spike. Height is dN/dln(w),
            #  which would theoretically be infinity in this case.
            x_coords = [weight_value]
            density = [dN / dlnw]
        else:
            # If not, fall back to the standard plot
            x_coords, density = super().compute_histogram()

        return x_coords, density


class EqualWeightDistributionPlot(WeightDistributionPlot):
    def plot(self, x_coords, density, color):
        if len(x_coords) == 1 and len(density) == 1:
            # Special case: all weights are equal
            self.ax.bar(
                x_coords[0],
                density[0],
                width=x_coords[0] * 0.1,
                color=color,
                log=True,
            )
        else:
            # If not, fall back to the standard plot
            super().plot(x_coords, density, color)


def histogram_plot(ax, x_label, df, col):
    histogram = StandardHistogram(df, col)
    plot = StandardHistogramPlot(histogram, ax, x_label)
    plot.create_plot()
    return plot


def comparative_histogram_plot(ax, x_label, df1, df2, col):
    h1 = StandardHistogram(df1, col)
    h2 = StandardHistogram(df2, col)
    plot = StandardHistogramPlot(h1 + h2, ax, x_label)
    plot.create_plot()
    return plot


def weight_distribution_plot(ax, df):
    histogram = EqualWeightHistogram(df)
    plot = EqualWeightDistributionPlot(histogram, ax)
    plot.create_plot()
    return plot


def comparative_weight_distribution_plot(ax, df1, df2):
    h1 = EqualWeightHistogram(df1)
    h2 = EqualWeightHistogram(df2)
    plot = EqualWeightDistributionPlot(h1 + h2, ax)
    plot.create_plot()
    return plot
