from river import ensemble, preprocessing, tree, drift, naive_bayes
from river.datasets.synth import RandomRBF
from generators.concept_drift import ConceptDriftStream
from generators.multi_class_drift import MultiClassDrift
from evaluators.multi_class_evaluator import MultiClassEvaluator
from generators.imbalance_generators import MultiClassImbalancedStream
from experiment import Experiment
from stream_generators import streams
from joblib import Parallel, delayed
import itertools
from drift_detectors import RDDM_M
from frouros.detectors.concept_drift.streaming.statistical_process_control.rddm import (
    RDDM,
)

models = [
    (
        "LB",
        ensemble.LeveragingBaggingClassifier(
            model=(tree.HoeffdingTreeClassifier()),
            n_models=10,
            seed=42,
        ),
    ),
    ("HT", ensemble.LeveragingBaggingClassifier(tree.HoeffdingTreeClassifier())),
    ("NB", naive_bayes.GaussianNB()),
]

dds = [
    ("ADWIN", drift.ADWIN()),
    ("DDM", drift.binary.DDM()),
    ("RDDM", RDDM_M()),
]


def task(stream, model, dd):
    stream_name, stream = stream
    model_name, model = model
    dd_name, dd = dd
    exp_name = "{}_{}_{}".format(model_name, dd_name, stream_name)
    print("Running {}...".format(exp_name))
    exp = Experiment(
        exp_name,
        "exp_output/",
        model,
        dd,
        stream,
    )

    exp.run()

    exp.save()


for model in models:
    out = Parallel(n_jobs=8)(
        delayed(task)(stream, model, dd)
        for stream, dd in itertools.product(streams, dds)
    )
