from river import ensemble, preprocessing, tree, drift
from river.datasets.synth import (
    RandomRBF,
    Hyperplane,
    Agrawal,
    RandomTree,
    Friedman,
    STAGGER,
    Sine,
)
from generators.concept_drift import ConceptDriftStream
from generators.multi_class_drift import MultiClassDrift
from evaluators.multi_class_evaluator import MultiClassEvaluator
from generators.imbalance_generators import MultiClassImbalancedStream
from experiment import Experiment


stable_rf_5 = ConceptDriftStream(
    MultiClassImbalancedStream(RandomRBF(42, 42, 5, 20, 50), [0.2, 0.2, 0.2, 0.2, 0.2]),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(
                RandomRBF(42, 42, 5, 20, 50), {0: 1, 1: 0, 2: 2, 3: 3, 4: 4}
            ),
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ),
        MultiClassImbalancedStream(
            MultiClassDrift(
                RandomRBF(42, 42, 5, 20, 50), {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}
            ),
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ),
        width=1,
        position=30000,
        size=100000,
    ),
    width=1,
    position=30000,
    size=100000,
)

stable_rf_10 = ConceptDriftStream(
    MultiClassImbalancedStream(
        RandomRBF(42, 42, 10, 20, 50),
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(
                RandomRBF(42, 42, 10, 20, 50),
                {0: 1, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 9, 7: 7, 8: 6, 9: 8},
            ),
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ),
        MultiClassImbalancedStream(
            MultiClassDrift(
                RandomRBF(42, 42, 10, 20, 50),
                {0: 1, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
            ),
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ),
        width=1,
        position=30000,
        size=100000,
    ),
    width=1,
    position=30000,
    size=100000,
)

stable_rt_5 = ConceptDriftStream(
    MultiClassImbalancedStream(
        RandomTree(
            seed_sample=42,
            seed_tree=42,
            n_cat_features=10,
            n_num_features=10,
            n_classes=5,
        ),
        [0.2, 0.2, 0.2, 0.2, 0.2],
    ),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(
                RandomTree(
                    seed_sample=42,
                    seed_tree=42,
                    n_cat_features=10,
                    n_num_features=10,
                    n_classes=5,
                ),
                {0: 1, 1: 0, 2: 2, 3: 3, 4: 4},
            ),
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ),
        MultiClassImbalancedStream(
            MultiClassDrift(
                RandomTree(
                    seed_sample=42,
                    seed_tree=42,
                    n_cat_features=10,
                    n_num_features=10,
                    n_classes=5,
                ),
                {0: 0, 1: 1, 2: 3, 3: 2, 4: 4},
            ),
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ),
        width=1,
        position=30000,
        size=100000,
    ),
    width=1,
    position=30000,
    size=100000,
)

stable_rt_10 = ConceptDriftStream(
    MultiClassImbalancedStream(
        RandomTree(
            seed_sample=42,
            seed_tree=42,
            n_cat_features=10,
            n_num_features=10,
            n_classes=10,
        ),
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ),
    ConceptDriftStream(
        MultiClassImbalancedStream(
            MultiClassDrift(
                RandomTree(
                    seed_sample=42,
                    seed_tree=42,
                    n_cat_features=10,
                    n_num_features=10,
                    n_classes=10,
                ),
                {0: 1, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 9, 7: 7, 8: 6, 9: 8},
            ),
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ),
        MultiClassImbalancedStream(
            MultiClassDrift(
                RandomTree(
                    seed_sample=42,
                    seed_tree=42,
                    n_cat_features=10,
                    n_num_features=10,
                    n_classes=10,
                ),
                {0: 1, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 9, 7: 7, 8: 6, 9: 8},
            ),
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ),
        width=1,
        position=30000,
        size=100000,
    ),
    width=1,
    position=30000,
    size=100000,
)


streams = [
    ("stable_rf_5", stable_rf_5),
    ("stable_rf_10", stable_rf_5),
    ("stable_rt_5", stable_rt_5),
    ("stable_rt_10", stable_rt_10),
]
