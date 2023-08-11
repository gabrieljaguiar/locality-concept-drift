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

"""
# NO_DRIFT_SCENARIO_BALANCED
for i in [2, 3, 5, 10]:
    streams.append(
        (
            "rbf_no_drift_{}_1:1".format(i),
            MultiClassImbalancedStream(
                RandomRBF(
                    42,
                    42,
                    n_classes=i,
                    n_features=2,
                    n_centroids=i * 2,
                    min_distance=0.3,
                    std_dev=0.1,
                ),
                getClassRatios(i, False),
            ),
        )
    )

    streams.append(
        (
            "rt_no_drift_{}_1:1".format(i),
            MultiClassImbalancedStream(
                RandomTreeMC(
                    42,
                    42,
                    n_classes=i,
                    n_num_features=2,
                    n_cat_features=0,
                    max_tree_depth=10,
                    first_leaf_level=10,
                ),
                getClassRatios(i, False),
            ),
        )
    )

    streams.append(
        (
            "hp_no_drift_{}_1:1".format(i),
            MultiClassImbalancedStream(
                HyperplaneMC(n_features=2, n_classes=i),
                getClassRatios(i, False),
            ),
        )
    )

# NO_DRIFT_SCENARIO_IMBALANCED
for i in [2, 3, 5, 10]:
    streams.append(
        (
            "rbf_no_drift_{}_1:{}".format(i, i),
            MultiClassImbalancedStream(
                RandomRBF(
                    42,
                    42,
                    n_classes=i,
                    n_features=2,
                    n_centroids=i * 2,
                    min_distance=0.3,
                    std_dev=0.1,
                ),
                [j / sum(range(1, i + 1)) for j in range(1, i + 1)],
            ),
        )
    )
    streams.append(
        (
            "rt_no_drift_{}_1:{}".format(i, 1),
            MultiClassImbalancedStream(
                RandomTreeMC(
                    42,
                    42,
                    n_classes=i,
                    n_num_features=2,
                    n_cat_features=0,
                    max_tree_depth=10,
                    first_leaf_level=10,
                ),
                [j / sum(range(1, i + 1)) for j in range(1, i + 1)],
            ),
        )
    )

    streams.append(
        (
            "hp_no_drift_{}_1:{}".format(i, i),
            MultiClassImbalancedStream(
                HyperplaneMC(n_features=2, n_classes=i),
                [j / sum(range(1, i + 1)) for j in range(1, i + 1)],
            ),
        )
    )



# INTRA-CLASS GLOBAL_NO_IMBALANCE
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
        base_stream_2.shift_cluster(
            i - 1
        )  # moving always the minority class if it is imbalance
        streams.append(
            (
                "intra_class_drift_global_shifting_cluster_{}_rbf_{}_1:1".format(ds, i),
                ConceptDriftStream(
                    MultiClassImbalancedStream(base_stream_1, getClassRatios(i, False)),
                    MultiClassImbalancedStream(base_stream_2, getClassRatios(i, False)),
                    width=ds,
                    position=SIZE / 2,
                    size=SIZE,
                ),
            )
        )

# INTRA-CLASS GLOBAL IMBALANCE
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
        base_stream_2.shift_cluster(
            i - 1
        )  # moving always the minority class if it is imbalance
        streams.append(
            (
                "intra_class_drift_global_shifting_cluster_{}_rbf_{}_1:{}".format(
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

streams_global = streams.copy()

# LOCAL_RBF = [SHIFT ONE OF THE CLUSTERS, ADD A NEW A CLUSTER]
# INTRA-CLASS LOCAL NO IMBALANCE
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
        base_stream_2.shift_cluster(
            i - 1, 0.5
        )  # moving always the minority class if it is imbalance
        streams.append(
            (
                "intra_class_drift_local_shifting_cluster_{}_rbf_{}_1:{}".format(
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


# INTRA-CLASS LOCAL NO IMBALANCE
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
        base_stream_2.add_cluster(
            i - 1
        )  # moving always the minority class if it is imbalance
        streams.append(
            (
                "intra_class_drift_local_emerging_cluster_{}_rbf_{}_1:{}".format(
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

# INTRA-CLASS LOCAL IMBALANCE
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
        base_stream_2.shift_cluster(
            i - 1, 0.5
        )  # moving always the minority class if it is imbalance
        streams.append(
            (
                "intra_class_drift_local_shifting_cluster_{}_rbf_{}_1:{}".format(
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

# INTRA-CLASS LOCAL IMBALANCE
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
        base_stream_2.add_cluster(
            i - 1
        )  # moving always the minority class if it is imbalance
        streams.append(
            (
                "intra_class_drift_local_emerging_cluster_{}_rbf_{}_1:{}".format(
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

"""


def save_csv(streams):
    name, stream = streams
    print("{}.csv".format(name))
    save_stream(stream, file="datasets/no_drift/{}.csv".format(name), size=SIZE)


if __name__ == "__main__":
    from utils.csv import save_stream

    from joblib import Parallel, delayed

    out = Parallel(n_jobs=8)(delayed(save_csv)(stream) for stream in streams)
