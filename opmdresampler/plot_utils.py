import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter


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


def customize_tick_labels(ax):
    for axis in [ax.yaxis, ax.xaxis]:
        axis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.tick_params(axis="both", which="major", labelsize=8)


def add_grid(ax):
    ax.grid(
        True,
        which="both",
        color="lightgray",
        ls="--",
        lw=0.5,
        alpha=0.6,
    )
