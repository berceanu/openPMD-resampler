from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np

from opmdresampler.plot_utils import add_grid, customize_tick_labels


class HistogramPlot(ABC):
    color = "royalblue"
    bins = 200
    weight_col = "weights"
    x_label = "x"
    y_label = "y"

    def __init__(
        self,
        ax,
        df,
    ):
        self.df = df
        self.ax = ax

    def default_plot_styling(self):
        return self.standard_plot_styling

    def plot(self, x_coords, density):
        self.ax.plot(
            x_coords,
            density,
            marker="o",
            linestyle="-",
            linewidth=0.4,
            color=self.color,
            markersize=1,
        )

    def standard_plot_styling(self):
        add_grid(self.ax)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        customize_tick_labels(self.ax)

    def right_axis_plot_styling(self):
        self.ax.set_xlabel(None)
        self.ax.set_ylabel(None)
        customize_tick_labels(self.ax)

    @abstractmethod
    def compute_histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def create_plot(
        self,
        plot_styling: Callable = None,
    ):
        if plot_styling is None:
            plot_styling = self.default_plot_styling()
        x_coords, density = self.compute_histogram()
        self.plot(x_coords, density)
        plot_styling()
        return self.ax

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise ValueError(f"Can only add another {type(self)} instance.")

        if self.weight_col != other.weight_col:
            raise ValueError("weight_col attributes are not the same.")
        if self.x_label != other.x_label:
            raise ValueError("x_label attributes are not the same.")
        if self.y_label != other.y_label:
            raise ValueError("y_label attributes are not the same.")

        other.ax = self.ax.twinx()
        other.color = "#FF7F0E"

        # Plot the data from the other instance on the new axis
        other.create_plot(other.right_axis_plot_styling)

        # Get the x-axis limits from both plots
        x_min_self, x_max_self = self.ax.get_xlim()
        x_min_other, x_max_other = other.ax.get_xlim()

        # Set the x-axis limits to the minimum and maximum x-values of both plots
        x_min = min(x_min_self, x_min_other)
        x_max = max(x_max_self, x_max_other)
        self.ax.set_xlim(x_min, x_max)
        other.ax.set_xlim(x_min, x_max)

        for obj in (self, other):
            obj.ax.tick_params(axis="y", labelcolor=obj.color)

        return self


class StandardHistogramPlot(HistogramPlot):
    def __init__(self, ax, df, col, x_label=None, y_label="Number of 'real' electrons"):
        super().__init__(ax, df)
        self.col = col
        self.x_label = x_label if x_label is not None else self.x_label
        self.y_label = y_label if y_label is not None else self.y_label

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


class WeightDistributionPlot(HistogramPlot):
    x_label = "w (weights)"
    y_label = "dN/dln(w)"

    def standard_plot_styling(self):
        add_grid(self.ax)
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)

    def right_axis_plot_styling(self):
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel(None)
        self.ax.set_ylabel(None)

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


class EqualWeightDistributionPlot(WeightDistributionPlot):
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

    def plot(self, x_coords, density):
        if len(x_coords) == 1 and len(density) == 1:
            # Special case: all weights are equal
            self.ax.bar(
                x_coords[0],
                density[0],
                width=x_coords[0] * 0.1,
                color=self.color,
                log=True,
            )
        else:
            # If not, fall back to the standard plot
            super().plot(x_coords, density)
