from .rbf import RandomRBFMC
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
    base_stream_2.remove_cluster(n_classes - 1, proportions=0.5)

    return ConceptDriftStream(
        MultiClassImbalancedStream(base_stream_1, getClassRatios(n_classes, imbalance)),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                base_stream_2, getClassRatios(n_classes, imbalance)
            ),
            MultiClassImbalancedStream(
                base_stream_1, getClassRatios(n_classes, imbalance)
            ),
            width=width,
            position=(SIZE / 2 - SIZE / 3),
            size=SIZE,
        ),
        width=width,
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
