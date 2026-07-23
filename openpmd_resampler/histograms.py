from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import numpy as np

from .plot_utils import add_grid, customize_tick_labels

try:
    import torch
except ImportError:  # torch only accelerates the histograms; numpy suffices
    torch = None


def _uniform_histogram(
    values: np.ndarray,
    number_of_intervals: int,
    weights: Optional[np.ndarray] = None,
    log_bins: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Histogram on uniformly spaced bins (linear, or uniform in log10 for
    log_bins) between the data min and max, like np.histogram on
    np.histogram_bin_edges / np.logspace edges. Aggregates on the GPU when
    PyTorch sees one (NVIDIA CUDA and AMD ROCm builds both expose it as
    "cuda") and falls back to NumPy otherwise; values on a bin boundary may
    land one bin off compared to np.histogram, which is invisible at plot
    resolution. Returns (bin_edges, counts).
    """
    if torch is not None and torch.cuda.is_available():
        # The bin edges only need the global min and max, which is cheap on
        # the CPU (log10 is monotonic, so the log-space edges follow from the
        # linear extrema); the aggregation then streams the data to the GPU in
        # chunks sized to the free memory, so columns larger than the GPU
        # still histogram there.
        low, high = float(values.min()), float(values.max())
        if log_bins:
            if low <= 0.0:
                raise ValueError("log_bins requires strictly positive values.")
            low, high = float(np.log10(low)), float(np.log10(high))
        if high == low:  # like np.histogram_bin_edges for constant data
            low, high = low - 0.5, high + 0.5
        free_bytes, _ = torch.cuda.mem_get_info()
        # values, weights and the long bin indices, plus transients.
        chunk = max(int(free_bytes * 0.25) // 24, 1_000_000)
        counts = torch.zeros(
            number_of_intervals,
            dtype=torch.long if weights is None else torch.float32,
            device="cuda",
        )
        for start in range(0, values.shape[0], chunk):
            # torch.tensor copies: pandas/numpy may hand out read-only arrays.
            tensor = torch.tensor(values[start : start + chunk], device="cuda")
            if log_bins:
                tensor = torch.log10(tensor)
            scaled = (tensor - low) / (high - low) * number_of_intervals
            indices = scaled.long().clamp_(0, number_of_intervals - 1)
            if weights is None:
                counts += torch.bincount(indices, minlength=number_of_intervals)
            else:
                # counts is float32; the weights column may be float64, and
                # index_add_ requires the source to match the destination dtype.
                counts.index_add_(
                    0,
                    indices,
                    torch.tensor(
                        weights[start : start + chunk],
                        dtype=torch.float32,
                        device="cuda",
                    ),
                )
        bin_edges = np.linspace(low, high, number_of_intervals + 1)
        if log_bins:
            bin_edges = 10.0**bin_edges
        return bin_edges, counts.cpu().numpy()

    if log_bins:
        if values.min() <= 0.0:
            raise ValueError("log_bins requires strictly positive values.")
        bin_edges = np.logspace(
            np.log10(values.min()), np.log10(values.max()), num=number_of_intervals + 1
        )
    else:
        bin_edges = np.histogram_bin_edges(values, bins=number_of_intervals)
    counts, bin_edges = np.histogram(values, bins=bin_edges, weights=weights)
    return bin_edges, counts


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
    bins = 1000
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
        bin_edges, counts = _uniform_histogram(
            self.df[self.col].to_numpy(),
            self.bins,
            weights=self.df[self.weight_col].to_numpy(),
        )

        density = counts

        x_coords = (bin_edges[:-1] + bin_edges[1:]) / 2

        return x_coords, density


class StandardHistogramPlot(HistogramPlot):
    y_label = "Number of 'real' particles"

    def plot(self, x_coords, density, color):
        self.ax.plot(
            x_coords,
            density,
            linestyle="-",
            linewidth=0.5,
            color=color,
            marker=",",
            markeredgewidth=0.0,
        )

class LogHistogramPlot(StandardHistogramPlot):
    def standard_plot_styling(self):
        self.ax.set_yscale("log")
        super().standard_plot_styling()

    def twin_axis_plot_styling(self):
        self.ax.set_yscale("log")
        super().twin_axis_plot_styling()

class WeightHistogram(Histogram):
    bins = 100

    def compute_histogram(self):
        # Logarithmic bins between the smallest and largest weight; self.bins
        # counts the edges, as np.logspace did before.
        bin_edges, counts = _uniform_histogram(
            self.df[self.weight_col].to_numpy(), self.bins - 1, log_bins=True
        )

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
        _, stemlines, _ = self.ax.stem(
            x_coords,
            density,
            linefmt=color,
            markerfmt=" ",
            basefmt=" ",
        )
        stemlines.set_linewidth(0.4)


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
