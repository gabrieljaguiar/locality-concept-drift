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
    **kwargs,
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
    **kwargs,
):
    if n_classes == len(classes_affected):
        return None
    imb_r = getClassRatios(n_classes, imbalance)

    base_stream_1 = get_base_rbf(n_classes, n_features)
    base_stream_2 = get_base_rbf(n_classes, n_features)

    for c in classes_affected:
        base_stream_2.remove_cluster(c, proportions=proportions)

    if proportions == 1:
        x = (1 / (n_classes - len(classes_affected))) - (1 / n_classes)
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
    **kwargs,
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
    **kwargs,
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
    **kwargs,
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


def swap_cluster(
    n_classes: int,
    n_features: int,
    width: int,
    classes_affected: list,
    proportions: float = 0.5,
    imbalance: bool = False,
    **kwargs,
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
    **kwargs,
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
    **kwargs,
):
    if n_classes == len(classes_affected):
        return None
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
    **kwargs,
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
        base_stream_2.prune_class(c, fraction=proportions)
        base_stream_2.create_new_node(c, fraction=proportions, overlap=False)

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
    **kwargs,
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
    **kwargs,
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


def generate_streams(
    n_classes: [list, int],
    n_features: [list, int],
    drift_width: [list, int],
    classes_affected: list,
    locality: str = None,
    generators: list = None,
    imbalance: bool = False,
    methods: list = None,
):
    """
    
    This function will generate a collection of data streams with Multi-Class concept drifts.

    Args:
        n_classes (list, int]): List of number of classes
        n_features (list, int]): List of number of features
        drift_width (list, int]): List of the drift width
        classes_affected (list): List of number of classes affected
        locality (str, optional): Locality of the drift. Either Global or Local
        generators (list, optional): List of generators to be used.
        imbalance (bool, optional): If imbalanced data streams are going to be generated.
        methods (list, optional): Specific difficulties to be generated.
    
    Returns:
        streams: List of generated streams

    """
    if generators == None:
        generators = ["rbf", "rt"]

    rbf_local = [
        "emerging_cluster",
        "reappearing_cluster",
        "splitting_cluster",
        "merging_cluster",
        "moving_cluster",
        "swap_cluster",
    ]
    rbf_global = [
        "reappearing_cluster",
        "moving_cluster",
        "splitting_cluster",
        "merging_cluster",
        "swap_cluster",
    ]
    rt_local = [
        "emerging_branch",
        "prune_regrowth_branch",
        "prune_growth_new_branch",
        "split_node",
        "swap_leaves",
    ]
    rt_global = [
        "prune_regrowth_branch",
        "prune_growth_new_branch",
        "split_node",
        "swap_leaves",
    ]
    if locality == "global":
        proportion = 1
    else:
        proportion = 0.3
    incremental_width = 2500

    if methods == None:
        methods = []
        for g in generators:
            if locality == "local" and g == "rt":
                methods = methods + rt_local
            if locality == "local" and g == "rbf":
                methods = methods + rbf_local
            if locality == "global" and g == "rt":
                methods = methods + rt_global
            if locality == "global" and g == "rbf":
                methods = methods + rbf_global
    functions = globals()
    # print(functions)

    streams = []
    for n_class in n_classes:
        idx = n_classes.index(n_class)
        spec_classes_affected = classes_affected[idx]
        for n_classes_affected in spec_classes_affected:
            if n_classes == n_classes_affected:
                list_classes_affected = [i for i in range(0, n_class)]
            else:
                list_classes_affected = [
                    i for i in range(n_class - n_classes_affected, n_class)
                ]

            for n_feat in n_features:
                for ds in drift_width:
                    for m in methods:
                        func = functions[m]
                        if imbalance == False:
                            stream_name = (
                                "multi_class_{}_{}_ds_{}_c_{}_ca_{}_f_{}_1_{}".format(
                                    locality,
                                    m,
                                    ds,
                                    n_class,
                                    n_classes_affected,
                                    n_feat,
                                    1,
                                )
                            )
                        else:
                            stream_name = (
                                "multi_class_{}_{}_ds_{}_c_{}_ca_{}_f_{}_1_{}".format(
                                    locality,
                                    m,
                                    ds,
                                    n_class,
                                    n_classes_affected,
                                    n_feat,
                                    n_class,
                                )
                            )

                        kwargs = {
                            "n_classes": n_class,
                            "n_features": n_feat,
                            "width": ds,
                            "incremental_width": incremental_width,
                            "imbalance": imbalance,
                            "proportions": proportion,
                            "classes_affected": list_classes_affected,
                        }

                        streams.append(
                            (
                                stream_name,
                                func(**kwargs),
                            )
                        )

    return streams
