from typing import List, Optional

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


class FigureLayout:
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
    savefig(filename: str) -> None:
        saves the figure to a file
    """

    def __init__(
        self,
        layout: List[int],
        W_px: Optional[int] = 1440,
        H_px: Optional[int] = 900,
        D_in: Optional[float] = 13.3,
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

        self.layout = layout
        self.W_px = W_px
        self.H_px = H_px
        self.D_in = D_in
        self.dpi = dpi if dpi else self.compute_dpi()
        self.fig, self.axs = self.create_figure_and_subplots()

    def compute_dpi(self) -> float:
        """Compute the DPI of a screen."""
        return (self.W_px**2 + self.H_px**2) ** 0.5 / self.D_in

    def get_ax(self, row, col):
        nrows, ncols = len(self.axs), len(self.axs[0])
        if row >= nrows or col >= ncols:
            raise IndexError(
                f"Axes index ({row, col}) is out of bounds for axs with size {nrows}x{ncols}."
            )
        return self.axs[row][col]

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
                row.append(ax)
            axs.append(row)

        # To make the plots fill the figure as much as possible
        fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)

        return fig, axs

    def savefig(self, filename: str) -> None:
        """
        Save the figure to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the figure to
        """

        self.fig.savefig(filename, dpi=self.dpi)
