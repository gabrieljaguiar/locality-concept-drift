from .rbf import RandomRBFMC
from .randomtree import RandomTreeMC
from .imbalance_generators import MultiClassImbalancedStream, getClassRatios
from .concept_drift import ConceptDriftStream

SIZE = 100000


def emerging_cluster(
    n_classes, n_features, width, proportions: float = 0.5, imbalance: bool = False
):
    n_centroids = max(n_features * n_classes, 24)

    base_stream_1 = RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.15,
        std_dev=0.1,
    )
    base_stream_2 = RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.15,
        std_dev=0.1,
    )

    for j in range(0, int(proportions * n_centroids)):
        base_stream_2.add_cluster(n_classes - 1)

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        MultiClassImbalancedStream(base_stream_2, getClassRatios(n_classes, imbalance)),
        width=width,
        position=SIZE / 2,
        size=SIZE,
    )


def reappearing_cluster(
    n_classes, n_features, width, proportions: float = 0.5, imbalance: bool = False
):
    n_centroids = max(n_features * n_classes, 24)

    imb_r = getClassRatios(n_classes, imbalance)

    base_stream_1 = RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.15,
        std_dev=0.1,
    )
    base_stream_2 = RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.15,
        std_dev=0.1,
    )
    base_stream_2.remove_cluster(n_classes - 1, proportions=proportions)

    if proportions == 1:
        x = (1 / n_classes) / (n_classes - 1)
        imb_r = [i + x for i in imb_r]
        imb_r[n_classes - 1] = 0

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        ConceptDriftStream(
            MultiClassImbalancedStream(base_stream_2, imb_r),
            MultiClassImbalancedStream(
                base_stream_1, getClassRatios(n_classes, imbalance)
            ),
            width=width,
            position=(SIZE / 2 - SIZE / 3),
            size=SIZE,
        ),
        width=1,
        position=SIZE / 3,
        size=SIZE,
    )


def splitting_cluster(
    n_classes: int,
    n_features: int,
    width: int,
    incremental_width: int = 1,
    proportions: float = 0.5,
    imbalance: bool = False,
):
    n_centroids = max(n_features * n_classes, 24)

    base_stream_1 = RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.15,
        std_dev=0.1,
    )
    base_stream_2 = RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.15,
        std_dev=0.1,
    )

    base_stream_2.split_cluster(
        n_classes - 1, n_classes - 1, width=incremental_width, proportion=proportions
    )

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        MultiClassImbalancedStream(base_stream_2, getClassRatios(n_classes, imbalance)),
        width=width,
        position=SIZE / 2,
        size=SIZE,
    )


def merging_cluster(
    n_classes: int,
    n_features: int,
    width: int,
    incremental_width: int = 1,
    proportions: float = 0.5,
    imbalance: bool = False,
):
    n_centroids = max(n_features * n_classes, 24)

    # class_ratios =

    base_stream_1 = RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.15,
        std_dev=0.1,
    )

    base_stream_2 = RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.15,
        std_dev=0.1,
    )

    base_stream_2.merge_cluster(
        n_classes - 1, n_classes - 1, width=incremental_width, proportion=proportions
    )
    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        MultiClassImbalancedStream(base_stream_2, getClassRatios(n_classes, imbalance)),
        width=width,
        position=SIZE / 2,
        size=SIZE,
    )


def moving_cluster(
    n_classes: int,
    n_features: int,
    width: int,
    incremental_width: int = 1,
    proportions: float = 0.5,
    imbalance: bool = False,
):
    n_centroids = max(n_features * n_classes, 24)

    # class_ratios =

    base_stream_1 = RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.15,
        std_dev=0.1,
    )

    base_stream_2 = RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.15,
        std_dev=0.1,
    )

    base_stream_2.incremental_moving(
        n_classes - 1, proportions=proportions, width=incremental_width
    )
    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        MultiClassImbalancedStream(base_stream_2, getClassRatios(n_classes, imbalance)),
        width=width,
        position=SIZE / 2,
        size=SIZE,
    )


def class_emerging(
    n_classes: int,
    n_features: int,
    width: int,
    imbalance: bool = False,
):
    n_centroids = max(n_features * n_classes, 24)

    # class_ratios =

    base_stream_1 = RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.15,
        std_dev=0.1,
    )
    imb_r = getClassRatios(n_classes - 1, imbalance)
    imb_r.append(0)
    base_stream_1.remove_cluster(n_classes - 1, proportions=1)

    base_stream_2 = RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.15,
        std_dev=0.1,
    )

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, imb_r),
        MultiClassImbalancedStream(base_stream_2, getClassRatios(n_classes, imbalance)),
        width=width,
        position=SIZE / 2,
        size=SIZE,
    )


def emerging_branch(
    n_classes: int,
    n_features: int,
    width: int,
    proportions: float = 0.5,
    imbalance: bool = False,
):
    base_stream_1 = RandomTreeMC(
        42,
        42,
        n_classes=n_classes,
        n_num_features=n_features,
        n_cat_features=0,
        n_categories_per_feature=0,
        max_tree_depth=10,
        first_leaf_level=9,
    )
    base_stream_2 = RandomTreeMC(
        42,
        42,
        n_classes=n_classes,
        n_num_features=n_features,
        n_cat_features=0,
        n_categories_per_feature=0,
        max_tree_depth=10,
        first_leaf_level=9,
    )
    base_stream_2.create_new_node(n_classes - 1, fraction=0.3)  # adding new branches

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        MultiClassImbalancedStream(base_stream_2, getClassRatios(n_classes, imbalance)),
        width=width,
        position=SIZE / 2,
        size=SIZE,
    )


def prune_regrowth_branch(
    n_classes: int,
    n_features: int,
    width: int,
    proportions: float = 0.5,
    imbalance: bool = False,
):
    imb_r = getClassRatios(n_classes, imbalance)

    base_stream_1 = RandomTreeMC(
        42,
        42,
        n_classes=n_classes,
        n_num_features=n_features,
        n_cat_features=0,
        n_categories_per_feature=0,
        max_tree_depth=10,
        first_leaf_level=9,
    )
    base_stream_2 = RandomTreeMC(
        42,
        42,
        n_classes=n_classes,
        n_num_features=n_features,
        n_cat_features=0,
        n_categories_per_feature=0,
        max_tree_depth=10,
        first_leaf_level=9,
    )

    if proportions == 1:
        x = (1 / n_classes) / (n_classes - 1)
        imb_r = [i + x for i in imb_r]
        imb_r[n_classes - 1] = 0

    base_stream_2.prune_class(n_classes - 1, fraction=proportions)

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        ConceptDriftStream(
            MultiClassImbalancedStream(base_stream_2, imb_r),
            MultiClassImbalancedStream(
                base_stream_1, getClassRatios(n_classes, imbalance)
            ),
            width=width,
            position=(SIZE / 2 - SIZE / 3),
            size=SIZE,
        ),
        width=1,
        position=SIZE / 3,
        size=SIZE,
    )


def prune_growth_new_branch(
    n_classes: int,
    n_features: int,
    width: int,
    proportions: float = 0.5,
    imbalance: bool = False,
):
    base_stream_1 = RandomTreeMC(
        42,
        42,
        n_classes=n_classes,
        n_num_features=n_features,
        n_cat_features=0,
        n_categories_per_feature=0,
        max_tree_depth=10,
        first_leaf_level=9,
    )
    base_stream_2 = RandomTreeMC(
        42,
        42,
        n_classes=n_classes,
        n_num_features=n_features,
        n_cat_features=0,
        n_categories_per_feature=0,
        max_tree_depth=10,
        first_leaf_level=9,
    )

    base_stream_2.prune_class(n_classes - 1, fraction=proportions)
    base_stream_2.create_new_node(n_classes - 1, fraction=proportions, overlap=False)

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        MultiClassImbalancedStream(base_stream_2, getClassRatios(n_classes, imbalance)),
        width=width,
        position=SIZE / 2,
        size=SIZE,
    )
