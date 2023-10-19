from generators.rbf import RandomRBFMC
from generators.imbalance_generators import MultiClassImbalancedStream, getClassRatios
from generators.concept_drift import ConceptDriftStream
from utils.csv import save_stream

SIZE = 100000

def get_base_rbf(n_classes, n_features):
    n_centroids = 2 * n_classes

    return RandomRBFMC(
        42,
        42,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        min_distance=0.3,
        std_dev=0.1,
    )

def moving_cluster_multi(
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

def moving_cluster(
    n_classes: int,
    n_features: int,
    width: int,
    incremental_width: int = 1,
    proportions: float = 0.5,
    imbalance: bool = False,
):

    # class_ratios =

    base_stream_1 = get_base_rbf(n_classes, n_features)

    base_stream_2 = get_base_rbf(n_classes, n_features)

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

single_class_local = moving_cluster(3,2,1, 5000, 0.5)
save_stream(single_class_local, file="single_class_local.csv", size=SIZE)


single_class_global = moving_cluster(3,2,1, 5000, 1)
save_stream(single_class_global, file="single_class_global.csv", size=SIZE)

multi_class_local = moving_cluster_multi(3,2,1,[0,1,2], 5000, 0.5)
save_stream(multi_class_local, file="multi_class_local.csv", size=SIZE)


multi_class_global = moving_cluster_multi(3,2,1,[0,1,2], 5000, 1)
save_stream(multi_class_global, file="multi_class_global.csv", size=SIZE)