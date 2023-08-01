from river.datasets.synth import RandomRBF
from river.datasets.synth.random_rbf import Centroid, random_index_based_on_weights
import random


class RBFMod(RandomRBF):
    def __init__(
        self,
        seed_model: int = None,
        seed_sample: int = None,
        n_classes: int = 2,
        n_features: int = 10,
        n_centroids: int = 50,
    ):
        super().__init__(seed_model, seed_sample, n_classes, n_features, n_centroids)

    def _generate_sample(self, rng_sample: random.Random):
        idx = random_index_based_on_weights(self.centroid_weights, rng_sample)

        current_centroid = self.centroids[idx]

        att_vals = dict()
        magnitude = 0.0
        for i in range(self.n_features):
            att_vals[i] = (rng_sample.random() * 2.0) - 1.0
            magnitude += att_vals[i] * att_vals[i]
        magnitude = magnitude**0.5
        desired_mag = rng_sample.gauss(0, 1) * current_centroid.std_dev
        scale = desired_mag / magnitude
        x = {
            i: current_centroid.centre[i] + att_vals[i] * scale
            for i in range(self.n_features)
        }
        y = current_centroid.class_label
        return x, y

    def _generate_centroids(self):
        """Generates centroids

        Sequentially creates all the centroids, choosing at random a center,
        a label, a standard deviation and a weight.

        """
        rng_model = random.Random(self.seed_model)
        self.centroids = []
        self.centroid_weights = []
        for i in range(self.n_centroids):
            self.centroids.append(Centroid())
            rand_centre = []
            for j in range(self.n_num_features):
                rand_centre.append(rng_model.random())
            self.centroids[i].centre = rand_centre
            self.centroids[i].class_label = rng_model.randint(0, self.n_classes - 1)
            self.centroids[i].std_dev = rng_model.random()
            self.centroid_weights.append(1)
