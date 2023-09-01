from typing import List, Optional, Tuple

import datashader as ds
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from datashader.mpl_ext import dsshow
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

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


# TODO: Refactor this class
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
        title: Optional[Tuple[str, str]] = None,
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
        self.title = title
        self.create_figure_and_subplots()

    @staticmethod
    def compute_dpi(W_px: int, H_px: int, D_in: float) -> float:
        """Compute the DPI of a screen."""
        return (W_px**2 + H_px**2) ** 0.5 / D_in

    def _get_ax(self, ax_position):
        nrows, ncols = len(self.axs), len(self.axs[0])

        if ax_position[0] >= nrows or ax_position[1] >= ncols:
            raise IndexError(
                f"Axes index {ax_position} is out of bounds for axs with size {nrows}x{ncols}."
            )

        return self.axs[ax_position[0]][ax_position[1]]

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

        # Add title to the figure
        if self.title is not None:
            self.fig.suptitle(self.title[0], y=0.92, color=self.title[1])

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

    def weight_distribution_plot(
        self,
        df: pd.DataFrame,
        ax_position: Optional[Tuple[int, int]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        weight_col: str = "weights",
        bins: int = 200,
        label: Optional[str] = None,
    ):
        if (ax_position is None and ax is None) or (
            ax_position is not None and ax is not None
        ):
            raise ValueError(
                "Either ax_position or ax should be specified, but not both."
            )

        if ax is None:
            using_right_y_axis = False
            color = "royalblue"
            ax = self._get_ax(ax_position)
        else:
            using_right_y_axis = True
            color = "#FF7F0E"  # orange

        if df[weight_col].nunique() == 1:  # All weights are the same
            weight_value = df[weight_col].iloc[0]  # take the first weight

            dN = len(df)
            dlnw = 1  # an infinitesimally small value for the natural logarithm of the same weight

            # Plot a single spike. Height is dN/dln(w), which would theoretically be infinity in this case.
            ax.bar(
                weight_value,
                dN / dlnw,
                width=weight_value * 0.05,
                color=color,
                log=True,
                label=label,
            )

            ax.set_xscale("log")
            ax.set_yscale("log")

            if not using_right_y_axis:
                ax.set_xlabel("w (weights)")
                ax.set_ylabel("dN/dln(w)")

                # Add fine grid lines
                ax.grid(
                    True,
                    which="both",
                    color="lightgray",
                    ls="--",
                    lw=0.5,
                    alpha=0.6,
                )

            ax.tick_params(axis="y", labelcolor=color)

            if not using_right_y_axis:
                ax.legend(loc="upper left", labelcolor=color)
            else:
                ax.legend(loc="upper center", labelcolor=color)

            return

        # Define the logarithmic bins
        log_bins = np.logspace(
            np.log10(df[weight_col].min()),
            np.log10(df[weight_col].max()),
            num=bins,
            base=10,
        )
        # Create histogram
        counts, bin_edges = np.histogram(df[weight_col], bins=log_bins)

        # The width of the bins in log scale is the difference in log-space of the bin edges
        bin_width = np.diff(np.log10(bin_edges))

        # Now we calculate the number of entries per bin divided by the width of the bin.
        # This is equivalent to the density of entries per bin in log scale.
        density = counts / bin_width

        # Set the x coordinates of the line plot.
        # Use the midpoint of each bin as x-coordinates.
        x_coords = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.plot(
            x_coords,
            density,
            marker="o",
            linestyle="-",
            linewidth=0.4,
            color=color,
            markersize=1,
            label=label,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        if not using_right_y_axis:
            ax.set_xlabel("w (weights)")
            ax.set_ylabel("dN/dln(w)")

            # Add fine grid lines
            ax.grid(
                True,
                which="both",
                color="lightgray",
                ls="--",
                lw=0.5,
                alpha=0.6,
            )

        ax.tick_params(axis="y", labelcolor=color)
        if not using_right_y_axis:
            ax.legend(loc="upper left", labelcolor=color)
        else:
            ax.legend(loc="upper center", labelcolor=color)

    def histogram_plot(
        self,
        df: pd.DataFrame,
        col: str,
        x_label: str,
        ax_position: Optional[Tuple[int, int]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        bins: int = 200,
        y_label: str = "Number of 'real' electrons",
        use_y_label: Optional[bool] = True,
        weight_col: str = "weights",
    ):
        if (ax_position is None and ax is None) or (
            ax_position is not None and ax is not None
        ):
            raise ValueError(
                "Either ax_position or ax should be specified, but not both."
            )

        # Calculate the histogram bin edges
        col_bins = np.histogram_bin_edges(df[col], bins=bins)

        # Set the x coordinates of the line plot.
        # Use the midpoint of each bin as x-coordinates.
        x_coords = (col_bins[:-1] + col_bins[1:]) / 2

        # Calculate the histogram
        counts, _ = np.histogram(
            df[col],
            bins=col_bins,
            weights=df[weight_col],
        )

        if ax is None:
            using_right_y_axis = False
            color = "royalblue"
            ax = self._get_ax(ax_position)
        else:
            using_right_y_axis = True
            color = "#FF7F0E"  # orange

        ax.plot(
            x_coords,
            counts,
            marker="o",
            linestyle="-",
            linewidth=0.4,
            color=color,
            markersize=1,
        )

        if not using_right_y_axis:
            ax.set_xlabel(x_label)
            if use_y_label:
                ax.set_ylabel(y_label)

            # Add fine grid lines
            ax.grid(
                True,
                which="both",
                color="lightgray",
                ls="--",
                lw=0.5,
                alpha=0.6,
            )

        # Customize tick labels
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="y", labelcolor=color)

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

        ax = self._get_ax(ax_position)

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

    def create_colorbar(
        self, norm, cbar_label: str = "Number of 'real' electrons"
    ) -> None:
        """Create a colorbar for the last artist used in a dsshow call."""
        self.fig.subplots_adjust(right=0.91)
        cax = self.fig.add_axes([0.92, 0.15, 0.025, 0.7])

        # Create a "fake" ScalarMappable with the colormap and norm
        sm = ScalarMappable(cmap=PURPLE_RABBIT, norm=norm)
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
