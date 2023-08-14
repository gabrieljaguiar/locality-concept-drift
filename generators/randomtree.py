from __future__ import annotations
from river.datasets.synth.random_tree import RandomTree, TreeNode

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
        self._prune_tree()
        self.leafs = self.get_leaf_nodes()

    def _generate_random_tree(self):
        """
        Generates the random tree, starting from the root node and following
        the constraints passed as parameters to the initializer.

        The tree is recursively generated, node by node, until it reaches the
        maximum tree depth.
        """
        rng_tree = random.Random(self.seed_tree)
        candidate_features = list(range(self.n_num_features + self.n_cat_features))
        min_numeric_values = [-1] * self.n_num_features
        max_numeric_values = [1] * self.n_num_features

        self.tree_root = self._generate_random_tree_node(
            0, candidate_features, min_numeric_values, max_numeric_values, rng_tree
        )

    def _collect_leaf_nodes(self, node, leafs):
        if node is not None:
            if len(node.children) == 0:
                leafs.append(node)

            for n in node.children:
                n.parent = node
                self._collect_leaf_nodes(n, leafs)

    def _prune_tree(self, fraction: float = 0.5):
        rng_tree = random.Random(self.seed_tree)
        to_be_removed = rng_tree.sample(
            population=self.leafs, k=int(fraction * len(self.leafs))
        )
        for leaf in to_be_removed:
            leaf.class_label = -1

    def create_new_node(self, class_1: int, fraction: float = 0.2):
        rng_tree = random.Random(self.seed_tree)
        leafs_to_be_changed = rng_tree.sample(
            population=[leaf for leaf in self.leafs if leaf.class_label == -1],
            k=int(fraction * len(self.leafs)),
        )
        for l in leafs_to_be_changed:
            if l.class_label == -1:
                l.class_label = class_1
        self.leafs = self.get_leaf_nodes()

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
            valid_instance = False
            while not valid_instance:
                try:
                    x = dict()
                    for feature in self.features_num:
                        x[feature] = rng_sample.uniform(-1, 1)
                    for feature in self.features_cat:
                        x[feature] = rng_sample.randint(
                            0, self.n_categories_per_feature - 1
                        )
                    y = self._classify_instance(self.tree_root, x)

                    valid_instance = y != -1
                except Exception as e:
                    valid_instance = False
            yield x, y
