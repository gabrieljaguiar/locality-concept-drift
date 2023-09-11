from __future__ import annotations
from river.datasets.synth import RandomRBF
from river.datasets.synth.random_rbf import Centroid, random_index_based_on_weights
import random
import math
from typing import List


class RandomRBFMC(RandomRBF):
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
        self.moving_centroids: list[MovingCentroid] = []
        self._generate_centroids()

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
                    rand_centre.append(self.rng_model.uniform(-1, 1))

            self.centroids[i].centre = rand_centre
            self.centroids[i].class_label = classes_assinged.pop(
                self.rng_model.randint(0, len(classes_assinged) - 1)
            )
            if len(classes_assinged) == 0:
                classes_assinged = [i for i in range(self.n_classes)]
            self.centroids[i].std_dev = self.std_dev
            self.centroid_weights.append(1)

    def _generate_sample(self, rng_sample: random.Random):
        idx = random_index_based_on_weights(self.centroid_weights, rng_sample)
        current_centroid = self.centroids[idx]

        moving_centers = [m.c for m in self.moving_centroids]
        if current_centroid in moving_centers:
            # print("moving centroid selected")
            self.moving_centroids[moving_centers.index(current_centroid)].update()
        att_vals = dict()
        magnitude = 0.0
        for i in range(self.n_features):
            att_vals[i] = (rng_sample.uniform(-1, 1) * 2.0) - 1.0
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

    def swap_clusters(self, class_1: int, class_2: int, proportions: float = 0.5):
        class_1_centroids = [c for c in self.centroids if c.class_label == class_1]
        class_2_centroids = [c for c in self.centroids if c.class_label == class_2]

        class_1_centroid = self.rng_model.sample(
            class_1_centroids, k=int(len(class_1_centroids) * proportions)
        )
        class_2_centroid = self.rng_model.sample(
            class_2_centroids, k=int(len(class_2_centroids) * proportions)
        )

        for c in class_1_centroid:
            c.class_label = class_2
        for c in class_2_centroid:
            c.class_label = class_1

    def remove_cluster(self, class_1: int, proportions: float = 0.5):
        class_1_centroids = [c for c in self.centroids if c.class_label == class_1]
        to_be_removed_clusters = self.rng_model.sample(
            class_1_centroids, k=int(len(class_1_centroids) * proportions)
        )
        for c in to_be_removed_clusters:
            index = self.centroids.index(c)
            self.centroids.remove(c)
            self.centroid_weights.pop(index)

    def add_cluster(self, class_1: int, weight: float = 1.0):
        self.centroids.append(Centroid())
        i = len(self.centroids) - 1
        rand_centre = None
        while not self._compute_nearest(rand_centre):
            rand_centre = []
            for j in range(self.n_num_features):
                rand_centre.append(self.rng_model.uniform(-1, 1))

        self.centroids[i].centre = rand_centre
        self.centroids[i].class_label = class_1
        self.centroids[i].std_dev = self.std_dev
        self.centroid_weights.append(weight)

    def shift_cluster(self, class_1: int, proportions: float = 1.0):
        class_centroids = [c for c in self.centroids if c.class_label == class_1]
        for i in range(int(len(class_centroids) * proportions)):
            self.add_cluster(class_1=class_1)
        to_be_removed_clusters = self.rng_model.sample(
            class_centroids, k=int(len(class_centroids) * proportions)
        )
        for c in to_be_removed_clusters:
            index = self.centroids.index(c)
            self.centroids.remove(c)
            self.centroid_weights.pop(index)

    def split_cluster(
        self,
        class_1: int,
        class_2: int,
        shift_mag: float = 0.3,
        width: int = 1,
        proportion: float = 0.5,
    ):
        class_centroids = [c for c in self.centroids if c.class_label == class_1]
        for j in range(0, int(len(class_centroids) * proportion)):
            c = self.rng_model.choice(class_centroids)
            class_centroids.remove(c)
            centroid = c.centre
            shift = self.rng_model.uniform(0.1, shift_mag)

            start_center = centroid.copy()
            center_1 = [(att + shift) for att in centroid]
            center_2 = [(att - shift) for att in centroid]

            index = self.centroids.index(c)
            self.centroids.remove(c)
            self.centroid_weights.pop(index)
            c1 = Centroid()
            c2 = Centroid()

            c1.centre = start_center
            c1.class_label = class_1
            c1.std_dev = self.std_dev
            self.centroids.append(c1)
            self.centroid_weights.append(1)

            self.moving_centroids.append(MovingCentroid(c1, c1.centre, center_1, width))

            c2.centre = start_center
            c2.class_label = class_2
            c2.std_dev = self.std_dev
            self.centroids.append(c2)
            self.centroid_weights.append(1)
            self.moving_centroids.append(MovingCentroid(c2, c2.centre, center_2, width))

    def merge_cluster(
        self, class_1: int, class_2: int, width: int = 1, proportion: float = 0.5
    ):
        class_centroids_1 = [c for c in self.centroids if c.class_label == class_1]
        if len(class_centroids_1) % 2 != 0:
            class_centroids_1.pop(0)

        class_centroids_2 = [c for c in self.centroids if c.class_label == class_2]
        if len(class_centroids_2) % 2 != 0:
            class_centroids_2.pop(0)

        n_merges = int(len(class_centroids_1) * proportion / 2)

        for j in range(0, n_merges):
            c_1 = self.rng_model.choice(class_centroids_1)
            class_centroids_1.remove(c_1)
            if class_1 == class_2:
                c_2 = self.rng_model.choice(class_centroids_1)
                class_centroids_1.remove(c_2)
            else:
                c_2 = self.rng_model.choice(class_centroids_2)
                class_centroids_2.remove(c_2)

            center = [
                (c_1.centre[i] + c_2.centre[i]) / 2 for i in range(0, len(c_1.centre))
            ]

            self.moving_centroids.append(MovingCentroid(c_1, c_1.centre, center, width))
            self.moving_centroids.append(MovingCentroid(c_2, c_2.centre, center, width))

    def incremental_moving(
        self, class_1: int, proportions: float = 1.0, width: int = 100
    ):
        class_centroids = [c for c in self.centroids if c.class_label == class_1]
        to_be_moved = self.rng_model.sample(
            class_centroids, k=int(len(class_centroids) * proportions)
        )

        # print(to_be_moved)
        for i in range(int(len(to_be_moved))):
            rand_centre = None
            while not self._compute_nearest(rand_centre):
                rand_centre = []
                for j in range(self.n_num_features):
                    rand_centre.append(self.rng_model.uniform(-1, 1))

            # print(rand_centre)

            self.moving_centroids.append(
                MovingCentroid(
                    to_be_moved[i], to_be_moved[i].centre, rand_centre, width
                )
            )

        # print(self.moving_centroids)

    def __iter__(self):
        rng_sample = random.Random(self.seed_sample)

        while True:
            x, y = self._generate_sample(rng_sample)
            yield x, y


class MovingCentroid:
    def __init__(self, c: Centroid, centre_1: list, centre_2: list, width: int) -> None:
        self.c = c
        self.centre_1 = centre_1.copy()
        self.centre_2 = centre_2
        self.width = width
        self.instanceCount = 0

    def update(self):
        try:
            factor = min(self.instanceCount / self.width, 1)
        except:
            factor = 0

        for i in range(0, len(self.centre_1)):
            self.c.centre[i] = (1 - factor) * self.centre_1[i] + (
                factor
            ) * self.centre_2[i]

        self.instanceCount += 1


# 0.99 -> 0
# 0.01 -> self.width
