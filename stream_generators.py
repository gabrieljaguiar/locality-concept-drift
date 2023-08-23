from river import ensemble, preprocessing, tree, drift
from river.datasets.synth import RandomTree
from generators import RandomRBF, RandomTreeMC, HyperplaneMC
from generators.concept_drift import ConceptDriftStream
from generators.multi_class_drift import MultiClassDrift
from evaluators.multi_class_evaluator import MultiClassEvaluator
from generators.imbalance_generators import MultiClassImbalancedStream
from experiment import Experiment
from random import Random


_rng = Random(42)
SIZE = 100000


def getClassRatios(n_classes: int, imbalance: bool = True):
    if not imbalance:
        return [1 / n_classes for i in range(n_classes)]
    else:
        proportions = [1] * n_classes
        proportions[len(proportions) - 1] = 1 / n_classes
        return [proportions[i] / sum(proportions) for i in range(0, n_classes)]


streams = []


# NO_DRIFT_SCENARIO_BALANCED
no_drift = []

for i in [2, 3, 5, 10]:
    for f in [2, 5, 10]:
        no_drift.append(
            (
                "no_drift/no_drift_rbf_f_{}_c_{}_1:1".format(f, i),
                MultiClassImbalancedStream(
                    RandomRBF(
                        42,
                        42,
                        n_classes=i,
                        n_features=f,
                        n_centroids=i * 2,
                        min_distance=0.3,
                        std_dev=0.1,
                    ),
                    getClassRatios(i, False),
                ),
            )
        )

        no_drift.append(
            (
                "no_drift/no_drift_rt_f_{}_c_{}_1:1".format(f, i),
                MultiClassImbalancedStream(
                    RandomTreeMC(
                        42,
                        42,
                        n_classes=i,
                        n_num_features=f,
                        n_cat_features=0,
                        max_tree_depth=10,
                        first_leaf_level=10,
                    ),
                    getClassRatios(i, False),
                ),
            )
        )

        no_drift.append(
            (
                "no_drift/no_drift_hyp_f_{}_c_{}_1:1".format(f, i),
                MultiClassImbalancedStream(
                    HyperplaneMC(n_features=f, n_classes=i),
                    getClassRatios(i, False),
                ),
            )
        )

# NO_DRIFT_SCENARIO_IMBALANCED
for i in [2, 3, 5, 10]:
    for f in [2, 5, 10]:
        no_drift.append(
            (
                "no_drift/no_drift_rbf_f_{}_c_{}_1:{}".format(f, i, i),
                MultiClassImbalancedStream(
                    RandomRBF(
                        42,
                        42,
                        n_classes=i,
                        n_features=f,
                        n_centroids=i * 2,
                        min_distance=0.3,
                        std_dev=0.1,
                    ),
                    [j / sum(range(1, i + 1)) for j in range(1, i + 1)],
                ),
            )
        )
        no_drift.append(
            (
                "no_drift/no_drift_rt_f_{}_c_{}_1:{}".format(f, i, i),
                MultiClassImbalancedStream(
                    RandomTreeMC(
                        42,
                        42,
                        n_classes=i,
                        n_num_features=f,
                        n_cat_features=0,
                        max_tree_depth=10,
                        first_leaf_level=10,
                    ),
                    [j / sum(range(1, i + 1)) for j in range(1, i + 1)],
                ),
            )
        )

        no_drift.append(
            (
                "no_drift/no_drift_hyp_f_{}_c_{}_1:{}".format(f, i, i),
                MultiClassImbalancedStream(
                    HyperplaneMC(n_features=f, n_classes=i),
                    [j / sum(range(1, i + 1)) for j in range(1, i + 1)],
                ),
            )
        )


intra_class = []
# INTRA-CLASS GLOBAL_NO_IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            base_stream_2.shift_cluster(
                i - 1
            )  # moving always the minority class if it is imbalance
            intra_class.append(
                (
                    "intra_class/global/intra_class_drift_shifting_cluster_rbf_ds_{}_f_{}_c_{}_1:1".format(
                        ds, f, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )
# INTRA-CLASS GLOBAL NO IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2.prune_class(i - 1, fraction=1)
            base_stream_2.create_new_node(i - 1, fraction=0.8)
            intra_class.append(
                (
                    "intra_class/global/intra_class_drift_branch_swap_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTRA-CLASS GLOBAL IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            base_stream_2.shift_cluster(
                i - 1
            )  # moving always the minority class if it is imbalance
            intra_class.append(
                (
                    "intra_class/global/intra_class_drift_shifting_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )
# INTRA-CLASS GLOBAL IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2.prune_class(i - 1, fraction=1)
            base_stream_2.create_new_node(i - 1, fraction=0.8)
            intra_class.append(
                (
                    "intra_class/global/intra_class_drift_branch_swap_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )


# INTRA-CLASS LOCAL NO IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            base_stream_2.shift_cluster(
                i - 1, 0.5
            )  # moving always the minority class if it is imbalance
            intra_class.append(
                (
                    "intra_class/local/intra_class_drift_shifting_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )
# INTRA-CLASS LOCAL NO IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            base_stream_2.add_cluster(
                i - 1
            )  # moving always the minority class if it is imbalance
            intra_class.append(
                (
                    "intra_class/local/intra_class_drift_emerging_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )
# INTRA-CLASS LOCAL IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            base_stream_2.shift_cluster(
                i - 1, 0.5
            )  # moving always the minority class if it is imbalance
            intra_class.append(
                (
                    "intra_class/local/intra_class_drift_shifting_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )
# INTRA-CLASS LOCAL IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            base_stream_2.add_cluster(
                i - 1
            )  # moving always the minority class if it is imbalance
            intra_class.append(
                (
                    "intra_class/local/intra_class_drift_emerging_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )


# INTRA-CLASS LOCAL NO IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2.create_new_node(i - 1)  # adding new branches
            intra_class.append(
                (
                    "intra_class/local/intra_class_drift_emerging_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2.prune_class(i - 1)  # adding new branches
            intra_class.append(
                (
                    "intra_class/local/intra_class_drift_pruning_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2.prune_class(i - 1, fraction=0.3)  # adding new branches
            base_stream_2.create_new_node(i - 1, fraction=0.3)
            intra_class.append(
                (
                    "intra_class/local/intra_class_drift_regrowth_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTRA-CLASS LOCAL IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2.create_new_node(i - 1)  # adding new branches
            intra_class.append(
                (
                    "intra_class/local/intra_class_drift_emerging_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2.prune_class(i - 1)  # adding new branches
            intra_class.append(
                (
                    "intra_class/local/intra_class_drift_pruning_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2.prune_class(i - 1, fraction=0.3)  # adding new branches
            base_stream_2.create_new_node(i - 1, fraction=0.3)
            intra_class.append(
                (
                    "intra_class/local/intra_class_drift_regrowth_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )


# INTRA-CLASS GLOBAL NO IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2.prune_class(i - 1, fraction=1)
            base_stream_2.create_new_node(i - 1, fraction=0.2, overlap=False)
            intra_class.append(
                (
                    "intra_class/global/intra_class_drift_switch_branches_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTRA-CLASS GLOBAL IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2.prune_class(i - 1, fraction=1)
            base_stream_2.create_new_node(i - 1, fraction=0.2, overlap=False)
            intra_class.append(
                (
                    "intra_class/global/intra_class_drift_switch_branches_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )


inter_class = []
# INTER-CLASS GLOBAL NO IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = RandomRBF(
                52,
                52,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            inter_class.append(
                (
                    "inter_class/global/inter_class_drift_distribution_change_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2 = RandomTreeMC(
                52,
                52,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            inter_class.append(
                (
                    "inter_class/global/inter_class_drift_distribution_change_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = HyperplaneMC(n_classes=i, n_features=f, seed=42)
            base_stream_2 = HyperplaneMC(n_classes=i, n_features=f, seed=52)
            inter_class.append(
                (
                    "inter_class/global/inter_class_drift_distribution_change_hyp_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            drift_key = {key: (key + 1) for key in range(0, i)}
            drift_key[i - 1] = 0
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = MultiClassDrift(
                RandomRBF(
                    42,
                    42,
                    n_classes=i,
                    n_features=f,
                    n_centroids=i * 2,
                    min_distance=0.3,
                    std_dev=0.1,
                ),
                driftKey=drift_key,
            )
            inter_class.append(
                (
                    "inter_class/global/inter_class_drift_class_shift_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            drift_key = {key: (key + 1) for key in range(0, i)}
            drift_key[i - 1] = 0
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = MultiClassDrift(
                RandomTreeMC(
                    42,
                    42,
                    n_classes=i,
                    n_num_features=f,
                    n_cat_features=0,
                    n_categories_per_feature=0,
                    max_tree_depth=10,
                    first_leaf_level=9,
                ),
                driftKey=drift_key,
            )

            inter_class.append(
                (
                    "inter_class/global/inter_class_drift_class_shift_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            drift_key = {key: (key + 1) for key in range(0, i)}
            drift_key[i - 1] = 0
            base_stream_1 = HyperplaneMC(n_classes=i, n_features=f, seed=42)

            base_stream_2 = MultiClassDrift(
                HyperplaneMC(n_classes=i, n_features=f, seed=42),
                driftKey=drift_key,
            )
            inter_class.append(
                (
                    "inter_class/global/inter_class_drift_class_shift_hyp_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS GLOBAL IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = RandomRBF(
                52,
                52,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )
            inter_class.append(
                (
                    "inter_class/global/inter_class_drift_distribution_change_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            base_stream_2 = RandomTreeMC(
                52,
                52,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )
            inter_class.append(
                (
                    "inter_class/global/inter_class_drift_distribution_change_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = HyperplaneMC(n_classes=i, n_features=f, seed=42)
            base_stream_2 = HyperplaneMC(n_classes=i, n_features=f, seed=52)
            inter_class.append(
                (
                    "inter_class/global/inter_class_drift_distribution_change_hyp_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            drift_key = {key: (key + 1) for key in range(0, i)}
            drift_key[i - 1] = 0
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = MultiClassDrift(
                RandomRBF(
                    42,
                    42,
                    n_classes=i,
                    n_features=f,
                    n_centroids=i * 2,
                    min_distance=0.3,
                    std_dev=0.1,
                ),
                driftKey=drift_key,
            )
            inter_class.append(
                (
                    "inter_class/global/inter_class_drift_class_shift_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            drift_key = {key: (key + 1) for key in range(0, i)}
            drift_key[i - 1] = 0
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = MultiClassDrift(
                RandomTreeMC(
                    42,
                    42,
                    n_classes=i,
                    n_num_features=f,
                    n_cat_features=0,
                    n_categories_per_feature=0,
                    max_tree_depth=10,
                    first_leaf_level=9,
                ),
                driftKey=drift_key,
            )

            inter_class.append(
                (
                    "inter_class/global/inter_class_drift_class_shift_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            drift_key = {key: (key + 1) for key in range(0, i)}
            drift_key[i - 1] = 0
            base_stream_1 = HyperplaneMC(n_classes=i, n_features=f, seed=42)

            base_stream_2 = MultiClassDrift(
                HyperplaneMC(n_classes=i, n_features=f, seed=42),
                driftKey=drift_key,
            )
            inter_class.append(
                (
                    "inter_class/global/inter_class_drift_class_shift_hyp_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL NO IMBALANCE Two classes switch completely
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            drift_key = {key: (key) for key in range(0, i)}
            drift_key[i - 1] = i - 2
            drift_key[i - 2] = i - 1
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = MultiClassDrift(
                RandomRBF(
                    42,
                    42,
                    n_classes=i,
                    n_features=f,
                    n_centroids=i * 2,
                    min_distance=0.3,
                    std_dev=0.1,
                ),
                driftKey=drift_key,
            )
            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_local_shift_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            drift_key = {key: (key) for key in range(0, i)}
            drift_key[i - 1] = i - 2
            drift_key[i - 2] = i - 1
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = MultiClassDrift(
                RandomTreeMC(
                    42,
                    42,
                    n_classes=i,
                    n_num_features=f,
                    n_cat_features=0,
                    n_categories_per_feature=0,
                    max_tree_depth=10,
                    first_leaf_level=9,
                ),
                driftKey=drift_key,
            )

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_local_shift_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            drift_key = {key: (key) for key in range(0, i)}
            drift_key[i - 1] = i - 2
            drift_key[i - 2] = i - 1
            base_stream_1 = HyperplaneMC(n_classes=i, n_features=f, seed=42)

            base_stream_2 = MultiClassDrift(
                HyperplaneMC(n_classes=i, n_features=f, seed=42),
                driftKey=drift_key,
            )
            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_local_shift_hyp_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL IMBALANCE Two classes switch completely
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            drift_key = {key: (key) for key in range(0, i)}
            drift_key[i - 1] = i - 2
            drift_key[i - 2] = i - 1
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = MultiClassDrift(
                RandomRBF(
                    42,
                    42,
                    n_classes=i,
                    n_features=f,
                    n_centroids=i * 2,
                    min_distance=0.3,
                    std_dev=0.1,
                ),
                driftKey=drift_key,
            )
            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_local_shift_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )


for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            drift_key = {key: (key) for key in range(0, i)}
            drift_key[i - 1] = i - 2
            drift_key[i - 2] = i - 1
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = MultiClassDrift(
                RandomTreeMC(
                    42,
                    42,
                    n_classes=i,
                    n_num_features=f,
                    n_cat_features=0,
                    n_categories_per_feature=0,
                    max_tree_depth=10,
                    first_leaf_level=9,
                ),
                driftKey=drift_key,
            )

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_local_shift_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )


for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            drift_key = {key: (key) for key in range(0, i)}
            drift_key[i - 1] = i - 2
            drift_key[i - 2] = i - 1
            base_stream_1 = HyperplaneMC(n_classes=i, n_features=f, seed=42)

            base_stream_2 = MultiClassDrift(
                HyperplaneMC(n_classes=i, n_features=f, seed=42),
                driftKey=drift_key,
            )
            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_local_shift_hyp_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL NO IMBALANCE Two classes switch partially
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2.swap_clusters(i - 2, i - 1)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_local_swap_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2.swap_leafs(i - 1, i - 2, fraction=0.3)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_local_swap_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL IMBALANCE Two classes switch partially
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2.swap_clusters(i - 2, i - 1)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_local_swap_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )


for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2.swap_leafs(i - 1, i - 2, fraction=0.3)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_local_swap_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL NO IMBALANCE Two classes clusters shift RBF
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2.shift_cluster(i - 1, proportions=0.5)
            base_stream_2.shift_cluster(i - 2, proportions=0.5)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_shifting_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL NO IMBALANCE new clusters from two classes emerge RBF
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2.add_cluster(i - 1, weight=1.5)
            base_stream_2.add_cluster(i - 2, weight=1.5)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_emerging_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL NO IMBALANCE clusters from one class split in two clusters from different classes RBF
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2.split_cluster(i - 1, i - 2, shift_mag=0.25)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_split_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL NO IMBALANCE pruning branch of two classes
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2.prune_class(i - 1, fraction=0.2)
            base_stream_2.prune_class(i - 2, fraction=0.2)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_pruning_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL NO IMBALANCE emerging branch of two classes
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2.create_new_node(i - 1, fraction=0.15)
            base_stream_2.create_new_node(i - 2, fraction=0.15)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_emerging_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL NO IMBALANCE regrowth branch of two classes,
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2.prune_class(i - 1, fraction=0.2)
            base_stream_2.create_new_node(i - 1, fraction=0.15)
            base_stream_2.prune_class(i - 2, fraction=0.2)
            base_stream_2.create_new_node(i - 2, fraction=0.15)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_regrowth_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL NO IMBALANCE leaf split into a new node of class 1 and class 2.
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2.split_node(i - 1, i - 2, fraction=0.5)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_split_node_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, 1
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, False)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, False)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL IMBALANCE Two classes clusters shift RBF
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2.shift_cluster(i - 1, proportions=0.5)
            base_stream_2.shift_cluster(i - 2, proportions=0.5)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_shifting_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL IMBALANCE new clusters from two classes emerge RBF
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2.add_cluster(i - 1, weight=1.5)
            base_stream_2.add_cluster(i - 2, weight=1.5)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_emerging_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL IMBALANCE clusters from one class split in two clusters from different classes RBF
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2 = RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=f,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            )

            base_stream_2.split_cluster(i - 1, i - 2, shift_mag=0.25)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_split_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL IMBALANCE pruning branch of two classes
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2.prune_class(i - 1, fraction=0.2)
            base_stream_2.prune_class(i - 2, fraction=0.2)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_pruning_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL IMBALANCE emerging branch of two classes
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2.create_new_node(i - 1, fraction=0.15)
            base_stream_2.create_new_node(i - 2, fraction=0.15)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_emerging_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL IMBALANCE regrowth branch of two classes,
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2.prune_class(i - 1, fraction=0.2)
            base_stream_2.create_new_node(i - 1, fraction=0.15)
            base_stream_2.prune_class(i - 2, fraction=0.2)
            base_stream_2.create_new_node(i - 2, fraction=0.15)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_regrowth_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )

# INTER-CLASS LOCAL IMBALANCE leaf split into a new node of class 1 and class 2.
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        for f in [2, 5, 10]:
            base_stream_1 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2 = RandomTreeMC(
                42,
                42,
                n_classes=i,
                n_num_features=f,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            )

            base_stream_2.split_node(i - 1, i - 2, fraction=0.5)

            inter_class.append(
                (
                    "inter_class/local/inter_class_drift_split_node_rt_ds_{}_f_{}_c_{}_1:{}".format(
                        ds, f, i, i
                    ),
                    ConceptDriftStream(
                        MultiClassImbalancedStream(
                            base_stream_1, getClassRatios(i, True)
                        ),
                        MultiClassImbalancedStream(
                            base_stream_2, getClassRatios(i, True)
                        ),
                        width=ds,
                        position=SIZE / 2,
                        size=SIZE,
                    ),
                )
            )


def save_csv(streams):
    name, stream = streams
    print("{}.csv".format(name))
    save_stream(stream, file="datasets/{}.csv".format(name), size=SIZE)


if __name__ == "__main__":
    from utils.csv import save_stream

    from joblib import Parallel, delayed

    # print([name for name, _ in intra_class])
    # _ = Parallel(n_jobs=16)(delayed(save_csv)(stream) for stream in no_drift)
    _ = Parallel(n_jobs=16)(delayed(save_csv)(stream) for stream in inter_class)
