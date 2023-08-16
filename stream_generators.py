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


def switch_classes(no_switch_dict: dict):
    class_1 = _rng.randint(1, len(no_switch_dict) - 1)
    class_2 = _rng.randint(1, len(no_switch_dict) - 1)

    while class_2 == class_1:
        class_2 = _rng.randint(0, len(no_switch_dict) - 1)

    aux = no_switch_dict[class_1]
    no_switch_dict[class_1] = no_switch_dict[class_2]
    no_switch_dict[class_2] = aux

    return no_switch_dict


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
                "no_drift/rbf_no_drift_f_{}_c_{}_1:1".format(f, i),
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
                "no_drift/rt_no_drift_f_{}_c_{}_1:1".format(f, i),
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
                "no_drift/hp_no_drift_f_{}_c_{}_1:1".format(f, i),
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
                "no_drift/rbf_no_drift_f_{}_c_{}_1:{}".format(f, i, i),
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
                "no_drift/rt_no_drift_f_{}_c_{}_1:{}".format(f, i, i),
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
                "no_drift/hp_no_drift_f_{}_c_{}_1:{}".format(f, i, i),
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
                    "intra_class/global/intra_class_drift_global_shifting_cluster_rbf_ds_{}_f_{}_c_{}_1:1".format(
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
                    "intra_class/global/intra_class_drift_global_branch_swap_rt_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/global/intra_class_drift_global_shifting_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/global/intra_class_drift_global_branch_swap_rt_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/local/intra_class_drift_local_shifting_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/local/intra_class_drift_local_emerging_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/local/intra_class_drift_local_shifting_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/local/intra_class_drift_local_emerging_cluster_rbf_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/local/intra_class_drift_local_emerging_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/local/intra_class_drift_local_pruning_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/local/intra_class_drift_local_regrowth_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/local/intra_class_drift_local_emerging_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/local/intra_class_drift_local_pruning_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/local/intra_class_drift_local_regrowth_branch_rt_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/global/intra_class_drift_global_switch_branches_rt_ds_{}_f_{}_c_{}_1:{}".format(
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
                    "intra_class/global/intra_class_drift_global_switch_branches_rt_ds_{}_f_{}_c_{}_1:{}".format(
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


"""
# INTER-CLASS GLOBAL NO IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2 = RandomRBF(
            52,
            52,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )
        streams.append(
            (
                "inter_class_drift_global_distribution_change_{}_rbf_{}_1:{}".format(
                    ds, i, 1
                ),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )
        base_stream_2 = RandomTreeMC(
            52,
            52,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )
        streams.append(
            (
                "inter_class_drift_global_distribution_change_{}_rt_{}_1:{}".format(
                    ds, i, 1
                ),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = HyperplaneMC(n_classes=i, n_features=2, seed=42)
        base_stream_2 = HyperplaneMC(n_classes=i, n_features=2, seed=52)
        streams.append(
            (
                "inter_class_drift_global_distribution_change_{}_hyp_{}_1:{}".format(
                    ds, i, 1
                ),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        drift_key = {key: (key + 1) for key in range(0, i)}
        drift_key[i - 1] = 0
        base_stream_1 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2 = MultiClassDrift(
            RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=2,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            ),
            driftKey=drift_key,
        )
        streams.append(
            (
                "inter_class_drift_class_shift_{}_rbf_{}_1:{}".format(ds, i, 1),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        drift_key = {key: (key + 1) for key in range(0, i)}
        drift_key[i - 1] = 0
        base_stream_1 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
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
                n_num_features=2,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            ),
            driftKey=drift_key,
        )

        streams.append(
            (
                "inter_class_drift_class_shift_{}_rt_{}_1:{}".format(ds, i, 1),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        drift_key = {key: (key + 1) for key in range(0, i)}
        drift_key[i - 1] = 0
        base_stream_1 = HyperplaneMC(n_classes=i, n_features=2, seed=42)

        base_stream_2 = MultiClassDrift(
            HyperplaneMC(n_classes=i, n_features=2, seed=42),
            driftKey=drift_key,
        )
        streams.append(
            (
                "inter_class_drift_class_shift_{}_hyp_{}_1:{}".format(ds, i, 1),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

# INTER-CLASS GLOBAL IMBALANCE
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2 = RandomRBF(
            52,
            52,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )
        streams.append(
            (
                "inter_class_drift_global_distribution_change_{}_rbf_{}_1:{}".format(
                    ds, i, i
                ),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, True)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, True)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )
        base_stream_2 = RandomTreeMC(
            52,
            52,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )
        streams.append(
            (
                "inter_class_drift_global_distribution_change_{}_rt_{}_1:{}".format(
                    ds, i, i
                ),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, True)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, True)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = HyperplaneMC(n_classes=i, n_features=2, seed=42)
        base_stream_2 = HyperplaneMC(n_classes=i, n_features=2, seed=52)
        streams.append(
            (
                "inter_class_drift_global_distribution_change_{}_hyp_{}_1:{}".format(
                    ds, i, i
                ),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, True)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, True)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        drift_key = {key: (key + 1) for key in range(0, i)}
        drift_key[i - 1] = 0
        base_stream_1 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2 = MultiClassDrift(
            RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=2,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            ),
            driftKey=drift_key,
        )
        streams.append(
            (
                "inter_class_drift_class_shift_{}_rbf_{}_1:{}".format(ds, i, i),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, True)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, True)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        drift_key = {key: (key + 1) for key in range(0, i)}
        drift_key[i - 1] = 0
        base_stream_1 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
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
                n_num_features=2,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            ),
            driftKey=drift_key,
        )

        streams.append(
            (
                "inter_class_drift_class_shift_{}_rt_{}_1:{}".format(ds, i, i),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, True)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, True)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        drift_key = {key: (key + 1) for key in range(0, i)}
        drift_key[i - 1] = 0
        base_stream_1 = HyperplaneMC(n_classes=i, n_features=2, seed=42)

        base_stream_2 = MultiClassDrift(
            HyperplaneMC(n_classes=i, n_features=2, seed=42),
            driftKey=drift_key,
        )
        streams.append(
            (
                "inter_class_drift_class_shift_{}_hyp_{}_1:{}".format(ds, i, i),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, True)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, True)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )
"""

"""
# INTER-CLASS LOCAL NO IMBALANCE Two classes switch completely
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        drift_key = {key: (key) for key in range(0, i)}
        drift_key[i - 1] = i - 2
        drift_key[i - 2] = i - 1
        base_stream_1 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2 = MultiClassDrift(
            RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=2,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            ),
            driftKey=drift_key,
        )
        streams.append(
            (
                "inter_class_drift_class_local_shift_{}_rbf_{}_1:{}".format(ds, i, 1),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        drift_key = {key: (key) for key in range(0, i)}
        drift_key[i - 1] = i - 2
        drift_key[i - 2] = i - 1
        base_stream_1 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
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
                n_num_features=2,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            ),
            driftKey=drift_key,
        )

        streams.append(
            (
                "inter_class_drift_class_local_shift_{}_rt_{}_1:{}".format(ds, i, 1),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        drift_key = {key: (key) for key in range(0, i)}
        drift_key[i - 1] = i - 2
        drift_key[i - 2] = i - 1
        base_stream_1 = HyperplaneMC(n_classes=i, n_features=2, seed=42)

        base_stream_2 = MultiClassDrift(
            HyperplaneMC(n_classes=i, n_features=2, seed=42),
            driftKey=drift_key,
        )
        streams.append(
            (
                "inter_class_drift_class_local_shift_{}_hyp_{}_1:{}".format(ds, i, 1),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

# INTER-CLASS LOCAL IMBALANCE Two classes switch completely
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        drift_key = {key: (key) for key in range(0, i)}
        drift_key[i - 1] = i - 2
        drift_key[i - 2] = i - 1
        base_stream_1 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2 = MultiClassDrift(
            RandomRBF(
                42,
                42,
                n_classes=i,
                n_features=2,
                n_centroids=i * 2,
                min_distance=0.3,
                std_dev=0.1,
            ),
            driftKey=drift_key,
        )
        streams.append(
            (
                "inter_class_drift_class_local_shift_{}_rbf_{}_1:{}".format(ds, i, i),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, True)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, True)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        drift_key = {key: (key) for key in range(0, i)}
        drift_key[i - 1] = i - 2
        drift_key[i - 2] = i - 1
        base_stream_1 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
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
                n_num_features=2,
                n_cat_features=0,
                n_categories_per_feature=0,
                max_tree_depth=10,
                first_leaf_level=9,
            ),
            driftKey=drift_key,
        )

        streams.append(
            (
                "inter_class_drift_class_local_shift_{}_rt_{}_1:{}".format(ds, i, i),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, True)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, True)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        drift_key = {key: (key) for key in range(0, i)}
        drift_key[i - 1] = i - 2
        drift_key[i - 2] = i - 1
        base_stream_1 = HyperplaneMC(n_classes=i, n_features=2, seed=42)

        base_stream_2 = MultiClassDrift(
            HyperplaneMC(n_classes=i, n_features=2, seed=42),
            driftKey=drift_key,
        )
        streams.append(
            (
                "inter_class_drift_class_local_shift_{}_hyp_{}_1:{}".format(ds, i, i),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, True)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, True)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )


# INTER-CLASS LOCAL NO IMBALANCE Two classes switch partially
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2.swap_clusters(i - 2, i - 1)

        streams.append(
            (
                "inter_class_drift_class_local_swap_{}_rbf_{}_1:{}".format(ds, i, 1),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

# INTER-CLASS LOCAL NO IMBALANCE Two classes switch partially
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )

        base_stream_2 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )

        base_stream_2.swap_leafs(i - 1, i - 2, fraction=0.3)

        streams.append(
            (
                "inter_class_drift_class_local_swap_{}_rt_{}_1:{}".format(ds, i, 1),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )


# INTER-CLASS LOCAL IMBALANCE Two classes switch partially
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2.swap_clusters(i - 2, i - 1)

        streams.append(
            (
                "inter_class_drift_class_local_swap_{}_rbf_{}_1:{}".format(ds, i, i),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, True)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, True)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

# INTER-CLASS LOCAL IMBALANCE Two classes switch partially
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )

        base_stream_2 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )

        base_stream_2.swap_leafs(i - 1, i - 2, fraction=0.3)

        streams.append(
            (
                "inter_class_drift_class_local_swap_{}_rt_{}_1:{}".format(ds, i, i),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, True)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, True)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )
"""

"""
# INTER-CLASS LOCAL NO IMBALANCE Two classes clusters shift RBF
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2.shift_cluster(i - 1, proportions=0.5)
        base_stream_2.shift_cluster(i - 2, proportions=0.5)

        streams.append(
            (
                "inter_class_drift_cluster_shift_{}_rbf_{}_1:{}".format(ds, i, 1),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

# INTER-CLASS LOCAL NO IMBALANCE new clusters from two classes emerge RBF
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2.add_cluster(i - 1, weight=1.5)
        base_stream_2.add_cluster(i - 2, weight=1.5)

        streams.append(
            (
                "inter_class_drift_emerging_cluster_{}_rbf_{}_1:{}".format(ds, i, 1),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

# INTER-CLASS LOCAL NO IMBALANCE clusters from one class split in two clusters from different classes RBF
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2 = RandomRBF(
            42,
            42,
            n_classes=i,
            n_features=2,
            n_centroids=i * 2,
            min_distance=0.3,
            std_dev=0.1,
        )

        base_stream_2.split_cluster(i - 1, i - 2, shift_mag=0.25)

        streams.append(
            (
                "inter_class_drift_split_cluster_{}_rbf_{}_1:{}".format(ds, i, 1),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

# INTER-CLASS LOCAL NO IMBALANCE pruning branch of two classes
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )

        base_stream_2 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )

        base_stream_2.prune_class(i - 1, fraction=0.2)
        base_stream_2.prune_class(i - 2, fraction=0.2)

        streams.append(
            (
                "inter_class_drift_class_pruning_branch_{}_rt_{}_1:{}".format(ds, i, 1),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

# INTER-CLASS LOCAL NO IMBALANCE emerging branch of two classes
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )

        base_stream_2 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )

        base_stream_2.create_new_node(i - 1, fraction=0.15)
        base_stream_2.create_new_node(i - 2, fraction=0.15)

        streams.append(
            (
                "inter_class_drift_class_emerging_branch_{}_rt_{}_1:{}".format(
                    ds, i, 1
                ),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

# INTER-CLASS LOCAL NO IMBALANCE regrowth branch of two classes,
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )

        base_stream_2 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )

        base_stream_2.prune_class(i - 1, fraction=0.2)
        base_stream_2.create_new_node(i - 1, fraction=0.15)
        base_stream_2.prune_class(i - 2, fraction=0.2)
        base_stream_2.create_new_node(i - 2, fraction=0.15)

        streams.append(
            (
                "inter_class_drift_class_emerging_branch_{}_rt_{}_1:{}".format(
                    ds, i, 1
                ),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

# INTER-CLASS LOCAL NO IMBALANCE leaf split into a new node of class 1 and class 2.
for i in [2, 3, 5, 10]:
    for ds in [1, 1000, 5000, 10000]:
        base_stream_1 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )

        base_stream_2 = RandomTreeMC(
            42,
            42,
            n_classes=i,
            n_num_features=2,
            n_cat_features=0,
            n_categories_per_feature=0,
            max_tree_depth=10,
            first_leaf_level=9,
        )

        base_stream_2.split_node(i - 1, i - 2, fraction=0.5)

        streams.append(
            (
                "inter_class_drift_split_node_{}_rt_{}_1:{}".format(ds, i, 1),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )
"""


def save_csv(streams):
    name, stream = streams
    print("{}.csv".format(name))
    save_stream(stream, file="datasets/{}.csv".format(name), size=SIZE)


if __name__ == "__main__":
    from utils.csv import save_stream

    from joblib import Parallel, delayed

    out = Parallel(n_jobs=8)(delayed(save_csv)(stream) for stream in no_drift)
