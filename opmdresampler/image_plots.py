from abc import ABC, abstractmethod
from typing import Callable

import datashader as ds
from datashader.mpl_ext import dsshow

from opmdresampler.plot_utils import PURPLE_RABBIT, customize_tick_labels


class DataShaderPlot(ABC):
    cmap = PURPLE_RABBIT
    weight_col = "weights"

    def __init__(self, ax, df, col_x, col_y, x_label, y_label) -> None:
        self.ax = ax
        self.df = df
        self.col_x = col_x
        self.col_y = col_y
        self.x_label = x_label
        self.y_label = y_label
        self.norm = "log"

    def default_plot_styling(self):
        return self.standard_plot_styling

    @abstractmethod
    def plot(self):
        """
        Abstract method for plotting data.
        """

    def add_colorbar(self, mappable, cbar_label):
        cbar = self.ax.figure.colorbar(mappable, ax=self.ax, label=cbar_label)
        return cbar

    def standard_plot_styling(self):
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label, labelpad=-2)
        customize_tick_labels(self.ax)

    def create_plot(
        self,
        plot_styling: Callable = None,
        add_cbar=True,
        cbar_label="Number of 'real' electrons",
    ):
        if plot_styling is None:
            plot_styling = self.default_plot_styling()
        mappable = self.plot()
        plot_styling()
        if add_cbar:
            self.add_colorbar(mappable, cbar_label)
        return self.ax


class StandardDataShaderPlot(DataShaderPlot):
    def plot(self):
        mappable = dsshow(
            self.df,
            ds.Point(self.col_x, self.col_y),
            ds.sum(self.weight_col),
            norm=self.norm,
            cmap=self.cmap,
            aspect="auto",
            ax=self.ax,
        )
        return mappable
