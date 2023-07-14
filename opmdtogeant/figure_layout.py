from typing import List, Optional, Tuple

import datashader as ds
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from datashader.mpl_ext import dsshow
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogFormatter, ScalarFormatter

# Screen resolution in pixels
DEFAULT_W_PX = 1440
DEFAULT_H_PX = 900

# Screen size in inches
D_IN = 13.3


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


class FigureCreator:
    """
    A class used to create matplotlib figures with a flexible layout.

    ...

    Attributes
    ----------
    W_px : int
        width of the screen in pixels
    H_px : int
        height of the screen in pixels
    dpi : float
        dots per inch of the screen
    layout : list
        list containing the number of columns in each row
    fig : matplotlib.figure.Figure
        the created figure
    axs : list of list of matplotlib.axes.Axes
        the subplot axes

    Methods
    -------
    compute_dpi(W_px: int, H_px: int, D_in: float) -> float:
        computes the DPI of a screen
    create_figure_and_subplots(pad: float = 3.0, h_pad: float = 3.0, w_pad: float = 3.0) -> None:
        creates a matplotlib figure and a grid of subplots
    save_figure(filename: str) -> None:
        saves the figure to a file
    """

    def __init__(
        self,
        layout: List[int],
        W_px: Optional[int] = DEFAULT_W_PX,
        H_px: Optional[int] = DEFAULT_H_PX,
        dpi: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        layout : list
            The list containing the number of columns in each row
        W_px : int, optional
            The width of the screen in pixels (default is DEFAULT_W_PX)
        H_px : int, optional
            The height of the screen in pixels (default is DEFAULT_H_PX)
        dpi : float, optional
            The dots per inch of the screen (default is computed via compute_dpi)
        """

        self.W_px = W_px
        self.H_px = H_px
        self.dpi = dpi if dpi else self.compute_dpi(W_px, H_px, D_IN)
        self.layout = layout
        self.create_figure_and_subplots()

    @staticmethod
    def compute_dpi(W_px: int, H_px: int, D_in: float) -> float:
        """Compute the DPI of a screen."""
        return (W_px**2 + H_px**2) ** 0.5 / D_in

    def create_figure_and_subplots(
        self, pad: float = 3.0, h_pad: float = 3.0, w_pad: float = 3.0
    ) -> None:
        """
        Create a matplotlib figure and a grid of subplots.

        Parameters
        ----------
        pad : float, optional
            Padding between the figure edge and the edges of subplots, as a fraction of the font size (default is 3.0)
        h_pad : float, optional
            Height padding between edges of adjacent subplots, as a fraction of the font size (default is 3.0)
        w_pad : float, optional
            Width padding between edges of adjacent subplots, as a fraction of the font size (default is 3.0)
        """

        # Convert pixels to inches for figure size
        W_in = self.W_px / self.dpi
        H_in = self.H_px / self.dpi

        # Create figure
        self.fig = Figure(figsize=(W_in, H_in), dpi=self.dpi)

        # Required for drawing onto the figure
        canvas = FigureCanvasAgg(self.fig)

        # Create a grid for the subplots
        nrows = len(self.layout)
        gs = GridSpec(nrows, 1, figure=self.fig)  # One column at the top level

        # Create subplots
        self.axs = []
        for i, ncols in enumerate(self.layout):
            # Create a new GridSpec for this row with the appropriate number of columns
            gs_i = gs[i].subgridspec(1, ncols)

            row = []
            for j in range(ncols):
                ax = self.fig.add_subplot(gs_i[0, j])
                row.append(ax)
            self.axs.append(row)

        # To make the plots fill the figure as much as possible
        self.fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)

    def shade_plot(
        self,
        ax_position: Tuple[int, int],
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        x_label: str,
        y_label: str,
        weight_col: str = "weights",
        norm: LogNorm = "log",
    ) -> None:
        """
        Create a datashader plot and shade it.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the data
        x_col : str
            The column name for the x axis
        y_col : str
            The column name for the y axis
        weight_col : str, optional
            The column name for the weights (default is "weights")
        ax_position : tuple
            The position of the axes on which to plot (row index, column index)
        x_label : str
            The label for the x axis
        y_label : str
            The label for the y axis
        """

        # TODO add out of bounds handling
        ax = self.axs[ax_position[0]][ax_position[1]]

        dsartist = dsshow(
            df,
            ds.Point(x_col, y_col),
            ds.sum(weight_col),
            norm=norm,
            cmap=PURPLE_RABBIT,
            aspect="auto",
            ax=ax,
        )

        # Customize x and y axes
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label, labelpad=-2)

        # Customize tick labels
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax.tick_params(axis="both", which="major", labelsize=8)

    def create_colorbar(self, cbar_label: str = "Number of 'real' electrons") -> None:
        """Create a colorbar for the last artist used in a dsshow call."""
        self.fig.subplots_adjust(right=0.91)
        cax = self.fig.add_axes([0.92, 0.15, 0.025, 0.7])

        # Create a "fake" ScalarMappable with the colormap and norm
        sm = ScalarMappable(cmap=PURPLE_RABBIT, norm=LogNorm(vmin=10, vmax=1e8))
        sm.set_array([])  # Make the image invisible

        # Now we can create the colorbar
        cbar = self.fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label(cbar_label)

    def save_figure(self, filename: str) -> None:
        """
        Save the figure to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the figure to
        """

        self.fig.savefig(filename, dpi=self.dpi)
