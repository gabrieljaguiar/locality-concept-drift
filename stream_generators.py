from river import ensemble, preprocessing, tree, drift
from river.datasets.synth import RandomRBF, RandomTree
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


no_class_imbalance_5 = [0.2 for i in range(5)]
no_class_imbalance_10 = [0.1 for i in range(10)]
no_class_imbalance_20 = [0.05 for i in range(15)]

fixed_imbalance_5 = [0.5, 0.2, 0.1, 0.1, 0.1]
fixed_imbalance_10 = [0.2, 0.15, 0.12, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.03]
fixed_imbalance_20 = [
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
]

no_switch_5 = dict((val, val) for val in range(5))
no_switch_10 = dict((val, val) for val in range(10))
no_switch_20 = dict((val, val) for val in range(20))


class_switch_5 = [no_switch_5.copy() for i in range(5)]
for i in range(1, 5):
    class_switch_5[i] = switch_classes(class_switch_5[i - 1].copy())

class_switch_10 = [no_switch_10.copy() for i in range(5)]
for i in range(1, 5):
    class_switch_10[i] = switch_classes(switch_classes(class_switch_10[i - 1].copy()))

class_switch_20 = [no_switch_20.copy() for i in range(5)]
for i in range(1, 5):
    class_switch_20[i] = switch_classes(switch_classes(class_switch_20[i - 1].copy()))

# circular_switch_5 = (
#    [0, 1, 2, 3, 4],
#    [1, 0, 2, 3, 4],
#    [1, 2, 0, 3, 4],
#    [1, 2, 3, 0, 4],
#    [1, 2, 3, 4, 0],
# )

# circular_switch_10 = (
#    [0, 1, 2, 3, 4],
#    [1, 0, 2, 3, 4],
#    [1, 2, 0, 3, 4],
#    [1, 2, 3, 0, 4],
#    [1, 2, 3, 4, 0],
# )

# circular_switch_20 = (
#    [0, 1, 2, 3, 4],
#    [1, 0, 2, 3, 4],
#    [1, 2, 0, 3, 4],
#    [1, 2, 3, 0, 4],
#    [1, 2, 3, 4, 0],
# )


no_imbalance_switching_rf_5_sudden = ConceptDriftStream(
    MultiClassImbalancedStream(RandomRBF(42, 42, 5, 20, 50), no_class_imbalance_5),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[1]),
            no_class_imbalance_5,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[2]),
                no_class_imbalance_5,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[3]),
                    no_class_imbalance_5,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[4]),
                    no_class_imbalance_5,
                ),
                width=1,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=1,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=1,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=1,
    position=SIZE / 5,
    size=SIZE,
)


imbalance_switching_rf_5_sudden = ConceptDriftStream(
    MultiClassImbalancedStream(RandomRBF(42, 42, 5, 20, 50), fixed_imbalance_5),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[1]),
            fixed_imbalance_5,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[2]),
                fixed_imbalance_5,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[3]),
                    fixed_imbalance_5,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[4]),
                    fixed_imbalance_5,
                ),
                width=1,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=1,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=1,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=1,
    position=SIZE / 5,
    size=SIZE,
)


no_imbalance_switching_rf_5_gradual = ConceptDriftStream(
    MultiClassImbalancedStream(RandomRBF(42, 42, 5, 20, 50), no_class_imbalance_5),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[1]),
            no_class_imbalance_5,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[2]),
                no_class_imbalance_5,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[3]),
                    no_class_imbalance_5,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[4]),
                    no_class_imbalance_5,
                ),
                width=500,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=500,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=500,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=500,
    position=SIZE / 5,
    size=SIZE,
)


imbalance_switching_rf_5_gradual = ConceptDriftStream(
    MultiClassImbalancedStream(RandomRBF(42, 42, 5, 20, 50), fixed_imbalance_5),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[1]),
            fixed_imbalance_5,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[2]),
                fixed_imbalance_5,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[3]),
                    fixed_imbalance_5,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), class_switch_5[4]),
                    fixed_imbalance_5,
                ),
                width=500,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=500,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=500,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=500,
    position=SIZE / 5,
    size=SIZE,
)


no_imbalance_switching_rt_5_sudden = ConceptDriftStream(
    MultiClassImbalancedStream(RandomTree(42, 42, 5, 10, 5), no_class_imbalance_5),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[1]),
            no_class_imbalance_5,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[2]),
                no_class_imbalance_5,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[3]),
                    no_class_imbalance_5,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[4]),
                    no_class_imbalance_5,
                ),
                width=1,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=1,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=1,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=1,
    position=SIZE / 5,
    size=SIZE,
)


imbalance_switching_rt_5_sudden = ConceptDriftStream(
    MultiClassImbalancedStream(RandomTree(42, 42, 5, 10, 5), fixed_imbalance_5),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[1]),
            fixed_imbalance_5,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[2]),
                fixed_imbalance_5,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[3]),
                    fixed_imbalance_5,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[4]),
                    fixed_imbalance_5,
                ),
                width=1,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=1,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=1,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=1,
    position=SIZE / 5,
    size=SIZE,
)


no_imbalance_switching_rt_5_gradual = ConceptDriftStream(
    MultiClassImbalancedStream(RandomTree(42, 42, 5, 10, 5), no_class_imbalance_5),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[1]),
            no_class_imbalance_5,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[2]),
                no_class_imbalance_5,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[3]),
                    no_class_imbalance_5,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[4]),
                    no_class_imbalance_5,
                ),
                width=500,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=500,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=500,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=500,
    position=SIZE / 5,
    size=SIZE,
)


imbalance_switching_rt_5_gradual = ConceptDriftStream(
    MultiClassImbalancedStream(RandomTree(42, 42, 5, 10, 5), fixed_imbalance_5),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[1]),
            fixed_imbalance_5,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[2]),
                fixed_imbalance_5,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[3]),
                    fixed_imbalance_5,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 5, 10, 5), class_switch_5[4]),
                    fixed_imbalance_5,
                ),
                width=500,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=500,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=500,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=500,
    position=SIZE / 5,
    size=SIZE,
)


no_imbalance_switching_rf_10_sudden = ConceptDriftStream(
    MultiClassImbalancedStream(RandomRBF(42, 42, 10, 20, 50), no_class_imbalance_10),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[1]),
            no_class_imbalance_10,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[2]),
                no_class_imbalance_10,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[3]),
                    no_class_imbalance_10,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[4]),
                    no_class_imbalance_10,
                ),
                width=1,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=1,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=1,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=1,
    position=SIZE / 5,
    size=SIZE,
)


imbalance_switching_rf_10_sudden = ConceptDriftStream(
    MultiClassImbalancedStream(RandomRBF(42, 42, 10, 20, 50), fixed_imbalance_10),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[1]),
            fixed_imbalance_10,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[2]),
                fixed_imbalance_10,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[3]),
                    fixed_imbalance_10,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[4]),
                    fixed_imbalance_10,
                ),
                width=1,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=1,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=1,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=1,
    position=SIZE / 5,
    size=SIZE,
)


no_imbalance_switching_rf_10_gradual = ConceptDriftStream(
    MultiClassImbalancedStream(RandomRBF(42, 42, 10, 20, 50), no_class_imbalance_10),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[1]),
            no_class_imbalance_10,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[2]),
                no_class_imbalance_10,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[3]),
                    no_class_imbalance_10,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[4]),
                    no_class_imbalance_10,
                ),
                width=500,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=500,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=500,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=500,
    position=SIZE / 5,
    size=SIZE,
)


imbalance_switching_rf_10_gradual = ConceptDriftStream(
    MultiClassImbalancedStream(RandomRBF(42, 42, 10, 20, 50), fixed_imbalance_10),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[1]),
            fixed_imbalance_10,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[2]),
                fixed_imbalance_10,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[3]),
                    fixed_imbalance_10,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomRBF(42, 42, 10, 20, 50), class_switch_10[4]),
                    fixed_imbalance_10,
                ),
                width=500,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=500,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=500,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=500,
    position=SIZE / 5,
    size=SIZE,
)


no_imbalance_switching_rt_10_sudden = ConceptDriftStream(
    MultiClassImbalancedStream(RandomTree(42, 42, 10, 10, 5), no_class_imbalance_10),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[1]),
            no_class_imbalance_10,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[2]),
                no_class_imbalance_10,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[3]),
                    no_class_imbalance_10,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[4]),
                    no_class_imbalance_10,
                ),
                width=1,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=1,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=1,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=1,
    position=SIZE / 5,
    size=SIZE,
)


imbalance_switching_rt_10_sudden = ConceptDriftStream(
    MultiClassImbalancedStream(RandomTree(42, 42, 10, 10, 5), fixed_imbalance_10),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[1]),
            fixed_imbalance_10,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[2]),
                fixed_imbalance_10,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[3]),
                    fixed_imbalance_10,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[4]),
                    fixed_imbalance_10,
                ),
                width=1,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=1,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=1,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=1,
    position=SIZE / 5,
    size=SIZE,
)


no_imbalance_switching_rt_10_gradual = ConceptDriftStream(
    MultiClassImbalancedStream(RandomTree(42, 42, 10, 10, 5), no_class_imbalance_10),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[1]),
            no_class_imbalance_10,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[2]),
                no_class_imbalance_10,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[3]),
                    no_class_imbalance_10,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[4]),
                    no_class_imbalance_10,
                ),
                width=500,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=500,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=500,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=500,
    position=SIZE / 5,
    size=SIZE,
)


imbalance_switching_rt_10_gradual = ConceptDriftStream(
    MultiClassImbalancedStream(RandomTree(42, 42, 10, 10, 5), fixed_imbalance_10),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[1]),
            fixed_imbalance_10,
        ),
        ConceptDriftStream(
            MultiClassImbalancedStream(
                MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[2]),
                fixed_imbalance_10,
            ),
            ConceptDriftStream(
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[3]),
                    fixed_imbalance_10,
                ),
                MultiClassImbalancedStream(
                    MultiClassDrift(RandomTree(42, 42, 10, 10, 5), class_switch_10[4]),
                    fixed_imbalance_10,
                ),
                width=500,
                position=SIZE / 5,
                size=SIZE,
            ),
            width=500,
            position=SIZE / 5,
            size=SIZE,
        ),
        width=500,
        position=SIZE / 5,
        size=SIZE,
    ),
    width=500,
    position=SIZE / 5,
    size=SIZE,
)

streams_5 = [
    (
        "no_imbalance_switching_rf_5_sudden_fix_majority",
        no_imbalance_switching_rf_5_sudden,
    ),
    (
        "no_imbalance_switching_rf_5_gradual_fix_majority",
        no_imbalance_switching_rf_5_gradual,
    ),
    ("imbalance_switching_rf_5_sudden_fix_majority", imbalance_switching_rf_5_sudden),
    ("imbalance_switching_rf_5_gradual_fix_majority", imbalance_switching_rf_5_gradual),
    (
        "no_imbalance_switching_rt_5_sudden_fix_majority",
        no_imbalance_switching_rt_5_sudden,
    ),
    (
        "no_imbalance_switching_rt_5_gradual_fix_majority",
        no_imbalance_switching_rt_5_gradual,
    ),
    ("imbalance_switching_rt_5_sudden_fix_majority", imbalance_switching_rt_5_sudden),
    ("imbalance_switching_rt_5_gradual_fix_majority", imbalance_switching_rt_5_gradual),
]
streams_10 = [
    (
        "no_imbalance_switching_rf_10_sudden_fix_majority",
        no_imbalance_switching_rf_10_sudden,
    ),
    (
        "no_imbalance_switching_rf_10_gradual_fix_majority",
        no_imbalance_switching_rf_10_gradual,
    ),
    ("imbalance_switching_rf_10_sudden_fix_majority", imbalance_switching_rf_10_sudden),
    (
        "imbalance_switching_rf_10_gradual_fix_majority",
        imbalance_switching_rf_10_gradual,
    ),
    (
        "no_imbalance_switching_rt_10_sudden_fix_majority",
        no_imbalance_switching_rt_10_sudden,
    ),
    (
        "no_imbalance_switching_rt_10_gradual_fix_majority",
        no_imbalance_switching_rt_10_gradual,
    ),
    ("imbalance_switching_rt_10_sudden_fix_majority", imbalance_switching_rt_10_sudden),
    (
        "imbalance_switching_rt_10_gradual_fix_majority",
        imbalance_switching_rt_10_gradual,
    ),
]

streams = streams_5 + streams_10
