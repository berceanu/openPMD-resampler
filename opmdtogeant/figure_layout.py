from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# Screen resolution in pixels
DEFAULT_W_PX = 1440
DEFAULT_H_PX = 900

# Screen size in inches
D_IN = 13.3


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

    Methods
    -------
    compute_dpi(W_px: int, H_px: int, D_in: float) -> float:
        computes the DPI of a screen
    create_figure_and_subplots(pad: float = 3.0, h_pad: float = 3.0, w_pad: float = 3.0) -> Figure:
        creates a matplotlib figure and a grid of subplots
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
        W_px : int, optional
            The width of the screen in pixels (default is DEFAULT_W_PX)
        H_px : int, optional
            The height of the screen in pixels (default is DEFAULT_H_PX)
        dpi : float, optional
            The dots per inch of the screen (default is computed via compute_dpi)
        layout : list
            The list containing the number of columns in each row
        """

        self.W_px = W_px
        self.H_px = H_px
        self.dpi = dpi if dpi else self.compute_dpi(W_px, H_px, D_IN)
        self.layout = layout

    @staticmethod
    def compute_dpi(W_px: int, H_px: int, D_in: float) -> float:
        """Compute the DPI of a screen."""
        return (W_px**2 + H_px**2) ** 0.5 / D_in

    def create_figure_and_subplots(
        self, pad: float = 3.0, h_pad: float = 3.0, w_pad: float = 3.0
    ) -> Figure:
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

        Returns
        -------
        fig
            A matplotlib.figure.Figure object with the desired layout of subplots
        axs
            A list of list of matplotlib.axes.Axes representing the subplot axes
        """

        # Convert pixels to inches for figure size
        W_in = self.W_px / self.dpi
        H_in = self.H_px / self.dpi

        # Create figure
        fig = Figure(figsize=(W_in, H_in), dpi=self.dpi)

        # Required for drawing onto the figure
        canvas = FigureCanvasAgg(fig)

        # Create a grid for the subplots
        nrows = len(self.layout)
        gs = GridSpec(nrows, 1, figure=fig)  # One column at the top level

        # Create subplots
        axs = []
        for i, ncols in enumerate(self.layout):
            # Create a new GridSpec for this row with the appropriate number of columns
            gs_i = gs[i].subgridspec(1, ncols)

            row = []
            for j in range(ncols):
                ax = fig.add_subplot(gs_i[0, j])
                # For now, just plot a simple line on each subplot
                ax.plot([0, 1, 2], [0, 1, 2])
                row.append(ax)
            axs.append(row)

        # To make the plots fill the figure as much as possible
        fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)

        return fig, axs
