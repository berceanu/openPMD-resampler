from abc import ABC, abstractmethod

import datashader as ds
import matplotlib.colors as mcolors
from datashader.mpl_ext import dsshow

from opmdresampler.utils import customize_tick_labels


def generate_custom_colormap(colors: list):
    """
    Generate a custom colormap from a given list of colors.

    :param colors: List of colors.
    :return: matplotlib colormap.
    """
    positions = [i / (len(colors) - 1) for i in range(len(colors))]
    return mcolors.LinearSegmentedColormap.from_list(
        "custom", list(zip(positions, colors))
    )


COLORS = ["black", "darkblue", "lightblue", "purple", "yellow"]
PURPLE_RABBIT = generate_custom_colormap(COLORS)


class DataShaderPlotStrategy(ABC):
    def __init__(self) -> None:
        self.norm = "log"
        self.cmap = PURPLE_RABBIT
        self.weight_col = "weights"

    def set_norm(self, norm):
        self.norm = norm

    def set_cmap(self, cmap):
        self.cmap = cmap

    def set_weight_col(self, weight_col):
        self.weight_col = weight_col

    @abstractmethod
    def plot(self, ax, df, col_x, col_y):
        """
        Abstract method for plotting data.

        :param ax: matplotlib axes object to draw the plot on.
        :param df: pandas DataFrame containing the data to plot.
        :param col_x: Name of the column in df to use for the x-axis.
        :param col_y: Name of the column in df to use for the y-axis.
        """

    def add_colorbar(self, ax, mappable, cbar_label):
        cbar = ax.figure.colorbar(mappable, ax=ax, label=cbar_label)
        return cbar

    def apply_plot_styling(self, ax, x_label, y_label):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label, labelpad=-2)
        customize_tick_labels(ax)

    def create_plot(
        self,
        df,
        col_x,
        col_y,
        ax,
        x_label,
        y_label,
        add_cbar=True,
        cbar_label="Number of 'real' electrons",
    ):
        mappable = self.plot(ax, df, col_x, col_y)
        self.apply_plot_styling(ax, x_label, y_label)
        if add_cbar:
            self.add_colorbar(ax, mappable, cbar_label)
        return ax


class StandardDataShaderPlot(DataShaderPlotStrategy):
    def plot(self, ax, df, col_x, col_y):
        mappable = dsshow(
            df,
            ds.Point(col_x, col_y),
            ds.sum(self.weight_col),
            norm=self.norm,
            cmap=self.cmap,
            aspect="auto",
            ax=ax,
        )
        return mappable
