import datashader as ds
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from datashader.mpl_ext import dsshow


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


def main():
    # Load the dataframe from the pickle file
    df = pd.read_pickle("reduced_dataframe.pkl")

    colors = ["black", "darkblue", "lightblue", "purple", "yellow"]
    cmap = generate_custom_colormap(colors)

    fig, ax = plt.subplots()
    dsartist = dsshow(
        df,
        ds.Point("position_x_um", "position_y_um"),
        ds.sum("weights"),
        norm="log",
        cmap=cmap,
        aspect="equal",
        ax=ax,
    )

    # Customize colorbar
    cbar = fig.colorbar(
        dsartist,
        ax=ax,
    )
    cbar.set_label("Number of 'real' electrons")

    # Customize x and y axes
    ax.set_xlabel("X Position (um)")
    ax.set_ylabel("Y Position (um)")
    ax.set_xlim(df["position_x_um"].min(), df["position_x_um"].max())
    ax.set_ylim(df["position_y_um"].min(), df["position_y_um"].max())

    fig.savefig("dsshow_log_weights.png")


if __name__ == "__main__":
    main()
