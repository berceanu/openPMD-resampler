import datashader as ds
import matplotlib.pyplot as plt
import pandas as pd


class DataAggregator:
    """Class to handle aggregation of data using datashader."""

    def __init__(self, dataframe, weights_column):
        self.dataframe = dataframe
        self.weights_column = weights_column

    def aggregate(self, x_column, y_column):
        """Aggregate data on the given columns."""

        # Create a new weighted column
        self.dataframe["weighted_y_column"] = (
            self.dataframe[y_column] * self.dataframe[self.weights_column]
        )
        # Create a canvas for plotting
        canvas = ds.Canvas()
        # Perform the aggregation
        agg = canvas.points(
            self.dataframe, x_column, "weighted_y_column", ds.mean(y_column)
        )
        # Remove the added weighted column to avoid side effects
        self.dataframe.drop(columns=["weighted_y_column"], inplace=True)
        return agg


class ParticleVisualizer:
    """Class to visualize particle data."""

    def __init__(self, dataframe, features, weights_column):
        self.dataframe = dataframe
        self.features = features
        self.weights_column = weights_column
        self.aggregator = DataAggregator(dataframe, weights_column)

    def _scatter_plot(self, x_column, y_column, ax):
        """Create scatter plot with the given columns."""
        agg = self.aggregator.aggregate(x_column, y_column)
        img = tf.shade(agg)
        img = tf.set_background(img, "white")
        ax.imshow(img.to_pil(), origin="lower")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)

    def _kde_plot(self, column, ax):
        """Create KDE plot with the given column."""

        self.dataframe[column].plot.kde(ax=ax)

    def _histogram(self, column, ax):
        """Create histogram with the given column."""

        self.dataframe[column].plot.hist(ax=ax)

    def visualize(self, output_filename):
        """Visualize data and save the plot to a file."""

        fig, axs = plt.subplots(len(self.features), len(self.features))

        for i, feature_i in enumerate(self.features):
            for j, feature_j in enumerate(self.features):
                ax = axs[i, j]
                if i == j:
                    self._histogram(feature_i, ax)
                elif i > j:
                    self._scatter_plot(feature_i, feature_j, ax)
                else:
                    self._kde_plot(feature_j, ax)

        plt.savefig(output_filename)
