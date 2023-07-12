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
    add_labels(x_labels: List[str], y_labels: List[str]) -> None:
        adds labels to the axes of the subplots
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
                # For now, just plot a simple line on each subplot
                ax.plot([0, 1, 2], [0, 1, 2])
                row.append(ax)
            self.axs.append(row)

        # To make the plots fill the figure as much as possible
        self.fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)

    def add_labels(self, x_labels: List[str], y_labels: List[str]) -> None:
        """
        Add labels to the axes of the subplots.

        Parameters
        ----------
        x_labels : list of str
            The x labels for the subplots
        y_labels : list of str
            The y labels for the subplots
        """

        label_idx = 0
        for row in self.axs:
            for ax in row:
                ax.set_xlabel(x_labels[label_idx])
                ax.set_ylabel(y_labels[label_idx])
                label_idx += 1

    def save_figure(self, filename: str) -> None:
        """
        Save the figure to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the figure to
        """

        self.fig.savefig(filename, dpi=self.dpi)
