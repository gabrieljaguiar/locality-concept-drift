from river.datasets.synth import RandomRBF
from river.datasets.synth.random_rbf import Centroid, random_index_based_on_weights
import random
from typing import List


class RandomRBF(RandomRBF):
    def __init__(
        self,
        seed_model: int = None,
        seed_sample: int = None,
        n_classes: int = 2,
        n_features: int = 10,
        n_centroids: int = 50,
        min_distance: float = 0.3,
        std_dev: float = 0.1,
    ):
        super().__init__(seed_model, seed_sample, n_classes, n_features, n_centroids)
        self.min_distance = min_distance
        self.std_dev = std_dev
        self.rng_model = random.Random(self.seed_model)
        self._generate_centroids()

    def __iter__(self):
        rng_sample = random.Random(self.seed_sample)

        while True:
            x, y = self._generate_sample(rng_sample)
            yield x, y

    def _compute_nearest(self, centroid: List[float]):
        import math

        if centroid == None:
            return False
        for c in self.centroids:
            if c.centre != None:
                dist = math.dist(centroid, c.centre)
                if dist < self.min_distance:
                    return False
        return True

    def _generate_centroids(self):
        """Generates centroids

        Sequentially creates all the centroids, choosing at random a center,
        a label, a standard deviation and a weight.

        """

        self.centroids = []
        self.centroid_weights = []
        classes_assinged = [i for i in range(self.n_classes)]
        for i in range(self.n_centroids):
            self.centroids.append(Centroid())
            rand_centre = None
            while not self._compute_nearest(rand_centre):
                rand_centre = []
                for j in range(self.n_num_features):
                    rand_centre.append(self.rng_model.random())

            self.centroids[i].centre = rand_centre
            self.centroids[i].class_label = classes_assinged.pop(
                self.rng_model.randint(0, len(classes_assinged) - 1)
            )
            if len(classes_assinged) == 0:
                classes_assinged = [i for i in range(self.n_classes)]
            self.centroids[i].std_dev = self.std_dev
            self.centroid_weights.append(1)

    def swap_clusters(self, class_1: int, class_2: int):
        class_1_centroids = [c for c in self.centroids if c.class_label == class_1]
        class_2_centroids = [c for c in self.centroids if c.class_label == class_2]

        class_1_centroid = self.rng_model.choice(class_1_centroids)
        class_2_centroid = self.rng_model.choice(class_2_centroids)

        class_1_centroid.class_label = class_2
        class_2_centroid.class_label = class_1

    def add_cluster(self, class_1: int):
        self.centroids.append(Centroid())
        i = len(self.centroids) - 1
        rand_centre = None
        while not self._compute_nearest(rand_centre):
            rand_centre = []
            for j in range(self.n_num_features):
                rand_centre.append(self.rng_model.random())

        self.centroids[i].centre = rand_centre
        self.centroids[i].class_label = class_1
        self.centroids[i].std_dev = self.std_dev
        self.centroid_weights.append(1)
