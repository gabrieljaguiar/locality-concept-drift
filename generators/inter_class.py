from .rbf import RandomRBFMC
from .randomtree import RandomTreeMC
from .imbalance_generators import MultiClassImbalancedStream, getClassRatios
from .concept_drift import ConceptDriftStream

SIZE = 100000


def get_base_rbf(n_classes, n_features):
    n_centroids = max(n_features * n_classes, 24)

    return RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.15,
        std_dev=0.1,
    )


def emerging_cluster(
    n_classes,
    n_features,
    width,
    classes_affected: list,
    proportions: float = 0.5,
    imbalance: bool = False,
):
    n_centroids = max(n_features * n_classes, 24)

    base_stream_1 = get_base_rbf(n_classes, n_features)
    base_stream_2 = get_base_rbf(n_classes, n_features)

    for c in classes_affected:
        for j in range(0, int(proportions * n_centroids / n_classes)):
            base_stream_2.add_cluster(c)

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        MultiClassImbalancedStream(base_stream_2, getClassRatios(n_classes, imbalance)),
        width=width,
        position=SIZE / 2,
        size=SIZE,
    )


def reappearing_cluster(
    n_classes,
    n_features,
    width,
    classes_affected: list,
    proportions: float = 0.5,
    imbalance: bool = False,
):
    imb_r = getClassRatios(n_classes, imbalance)

    base_stream_1 = get_base_rbf(n_classes, n_features)
    base_stream_2 = get_base_rbf(n_classes, n_features)

    for c in classes_affected:
        base_stream_2.remove_cluster(c, proportions=proportions)

    if proportions == 1:
        x = (1 / (n_classes - len(classes_affected))) - (1 / n_classes)
        print(x)
        imb_r = [i + x for i in imb_r]
        for c in classes_affected:
            imb_r[c] = 0

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
    classes_affected: list,
    incremental_width: int = 1,
    proportions: float = 0.5,
    imbalance: bool = False,
):
    n_centroids = max(n_features * n_classes, 24)

    base_stream_1 = get_base_rbf(n_classes, n_features)
    base_stream_2 = get_base_rbf(n_classes, n_features)
    for c in classes_affected:
        base_stream_2.split_cluster(
            c, c, width=incremental_width, proportion=proportions
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
    classes_affected: list,
    incremental_width: int = 1,
    proportions: float = 0.5,
    imbalance: bool = False,
):
    base_stream_1 = get_base_rbf(n_classes, n_features)

    base_stream_2 = get_base_rbf(n_classes, n_features)

    for c in classes_affected:
        base_stream_2.merge_cluster(
            c, c, width=incremental_width, proportion=proportions
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
    classes_affected: list,
    incremental_width: int = 1,
    proportions: float = 0.5,
    imbalance: bool = False,
):
    n_centroids = max(n_features * n_classes, 24)

    base_stream_1 = get_base_rbf(n_classes, n_features)

    base_stream_2 = get_base_rbf(n_classes, n_features)

    for c in classes_affected:
        base_stream_2.incremental_moving(
            c, proportions=proportions, width=incremental_width
        )

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        MultiClassImbalancedStream(base_stream_2, getClassRatios(n_classes, imbalance)),
        width=width,
        position=SIZE / 2,
        size=SIZE,
    )


def swapping_cluster(
    n_classes: int,
    n_features: int,
    width: int,
    classes_affected: list,
    proportions: float = 0.5,
    imbalance: bool = False,
):
    n_centroids = max(n_features * n_classes, 24)

    base_stream_1 = get_base_rbf(n_classes, n_features)

    base_stream_2 = get_base_rbf(n_classes, n_features)

    for i in range(0, len(classes_affected) - 1):
        base_stream_2.swap_clusters(
            classes_affected[i], classes_affected[i + 1], proportions=proportions
        )

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        MultiClassImbalancedStream(base_stream_2, getClassRatios(n_classes, imbalance)),
        width=width,
        position=SIZE / 2,
        size=SIZE,
    )


def emerging_branch(
    n_classes: int,
    n_features: int,
    width: int,
    classes_affected: list,
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
    for c in classes_affected:
        base_stream_2.create_new_node(c, fraction=proportions)  # adding new branches

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
    classes_affected: list,
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
        x = (1 / (n_classes - len(classes_affected))) - (1 / n_classes)
        print(x)
        imb_r = [i + x for i in imb_r]
        for c in classes_affected:
            imb_r[c] = 0

    for c in classes_affected:
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
    classes_affected: list,
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
    for c in classes_affected:
        base_stream_2.prune_class(n_classes - 1, fraction=proportions)
        base_stream_2.create_new_node(
            n_classes - 1, fraction=proportions, overlap=False
        )

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        MultiClassImbalancedStream(base_stream_2, getClassRatios(n_classes, imbalance)),
        width=width,
        position=SIZE / 2,
        size=SIZE,
    )


def split_node(
    n_classes: int,
    n_features: int,
    width: int,
    classes_affected: list,
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

    for i in range(0, len(classes_affected) - 1):
        base_stream_2.split_node(
            classes_affected[i], classes_affected[i + 1], fraction=proportions
        )

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        MultiClassImbalancedStream(base_stream_2, getClassRatios(n_classes, imbalance)),
        width=width,
        position=SIZE / 2,
        size=SIZE,
    )


def swap_leaves(
    n_classes: int,
    n_features: int,
    width: int,
    classes_affected: list,
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

    for i in range(0, len(classes_affected) - 1):
        base_stream_2.swap_leafs(
            classes_affected[i], classes_affected[i + 1], fraction=proportions
        )

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        MultiClassImbalancedStream(base_stream_2, getClassRatios(n_classes, imbalance)),
        width=width,
        position=SIZE / 2,
        size=SIZE,
    )
