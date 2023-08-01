from river import ensemble, preprocessing, tree
from river.datasets.synth import RandomRBF
from generators.concept_drift import ConceptDriftStream
from generators.multi_class_drift import MultiClassDrift
from evaluators.multi_class_evaluator import MultiClassEvaluator
from generators.imbalance_generators import MultiClassImbalancedStream


model = ensemble.LeveragingBaggingClassifier(
    model=(preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()),
    n_models=10,
    seed=42,
)

model = tree.HoeffdingTreeClassifier()

stream = ConceptDriftStream(
    MultiClassImbalancedStream(RandomRBF(42, 42, 5, 20, 50), [0.4, 0.2, 0.2, 0.1, 0.1]),
    MultiClassImbalancedStream(
        MultiClassDrift(RandomRBF(42, 42, 5, 20, 50), {0: 1, 1: 0, 2: 2, 3: 3, 4: 4}),
        [0.4, 0.2, 0.2, 0.1, 0.1],
    ),
    width=1,
    position=50000,
    size=100000,
)

evaluator = MultiClassEvaluator(500, 5)

for i, (x, y) in enumerate(stream.take(100000)):
    if i > 200:
        evaluator.addResult((x, y), model.predict_proba_one(x))
    # print(y)
    model.learn_one(x, y)

    if (i + 1) % 500 == 0:
        print("{} - {}".format(i, evaluator.getGMean()))
        print("{} - {}".format(i, evaluator.getClassRecall(classIdx=3)))
        print("{} - {}".format(i, evaluator.getClassRecall(classIdx=4)))
