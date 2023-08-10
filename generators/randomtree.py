from __future__ import annotations
from river.datasets.synth import RandomTree
import random


class RandomTreeMC(RandomTree):
    def __init__(
        self,
        seed_tree: int | None = None,
        seed_sample: int | None = None,
        n_classes: int = 2,
        n_num_features: int = 5,
        n_cat_features: int = 5,
        n_categories_per_feature: int = 5,
        max_tree_depth: int = 5,
        first_leaf_level: int = 3,
        fraction_leaves_per_level: float = 0.15,
    ):
        super().__init__(
            seed_tree,
            seed_sample,
            n_classes,
            n_num_features,
            n_cat_features,
            n_categories_per_feature,
            max_tree_depth,
            first_leaf_level,
            fraction_leaves_per_level,
        )
        self._generate_random_tree()
        self.leafs = self.get_leaf_nodes()

    def _collect_leaf_nodes(self, node, leafs):
        if node is not None:
            if len(node.children) == 0:
                leafs.append(node)

            for n in node.children:
                n.parent = node
                self._collect_leaf_nodes(n, leafs)

    def get_leaf_nodes(self):
        leafs = []
        self._collect_leaf_nodes(self.tree_root, leafs)
        return leafs

    def swap_leafs(self, class_1: int, class_2: int, fraction: float = 0.50):
        rng_sample = random.Random(self.seed_sample)
        class_1_leafs = [l for l in self.leafs if l.class_label == class_1]
        class_2_leafs = [l for l in self.leafs if l.class_label == class_2]

        swap_class_1 = rng_sample.choices(
            class_1_leafs, k=int(fraction * len(class_1_leafs))
        )
        swap_class_2 = rng_sample.choices(
            class_2_leafs, k=int(fraction * len(class_1_leafs))
        )

        for i in range(len(swap_class_1)):
            swap_class_1[i].class_label = class_2
            swap_class_2[i].class_label = class_1

    def __iter__(self):
        rng_sample = random.Random(self.seed_sample)

        # Randomly generate features, and then classify the resulting instance.
        while True:
            x = dict()
            for feature in self.features_num:
                x[feature] = rng_sample.random()
            for feature in self.features_cat:
                x[feature] = rng_sample.randint(0, self.n_categories_per_feature - 1)
            y = self._classify_instance(self.tree_root, x)
            yield x, y
