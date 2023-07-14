class ParticleVisualizer:
    def __init__(self, data, features, weights_column):
        self.data = data
        self.features = features
        self.weights_column = weights_column
        self.aggregator = ParticleAggregator(data, weights_column)



    def visualize(self, filename):
        num_features = len(self.features)
        fig = plt.figure(figsize=(4 * num_features, 4 * num_features))
        gs = GridSpec(num_features, num_features, figure=fig)

        for i, feature_i in enumerate(self.features):
            for j, feature_j in enumerate(self.features):
                ax = fig.add_subplot(gs[i, j])
                else:
                    self._scatter_plot(feature_j, feature_i, ax)

