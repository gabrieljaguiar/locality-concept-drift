from river import ensemble, preprocessing, tree, drift, naive_bayes
from river.datasets.synth import RandomRBF
from generators.concept_drift import ConceptDriftStream
from generators.multi_class_drift import MultiClassDrift
from evaluators.multi_class_evaluator import MultiClassEvaluator
from generators.imbalance_generators import MultiClassImbalancedStream
from experiment import Experiment

# from stream_generators import streams
from joblib import Parallel, delayed
import itertools
from drift_detectors import RDDM_M

from drift_detectors import RDDM_M, GMA_M, EDDM_M, STEPD_M, ECDDWT_M
from drift_detectors import GeometricMovingAverageConfig, ECDDWTConfig, EDDMConfig, RDDMConfig, STEPDConfig
from drift_detectors import MCADWIN
from river.stream import iter_csv
from glob import glob
import os
from utils.csv import CSVStream

models = [
    # (
    #    "LB",
    #    ensemble.LeveragingBaggingClassifier(
    #        model=(tree.HoeffdingTreeClassifier()),
    #        n_models=10,
    #        seed=42,
    #    ),
    # ),
    ("HT", tree.HoeffdingTreeClassifier()),
    # ("NB", naive_bayes.GaussianNB()),
]


dds = [
    #("ADWIN", drift.ADWIN()),
    #("PageHinkley", drift.PageHinkley()),
    ##("HDDM", drift.binary.HDDM_W()),
    #("KSWIN", drift.KSWIN()),
    #("DDM", drift.binary.DDM()),
    #("RDDM", RDDM_M(RDDMConfig())),
    ("STEPD", STEPD_M(STEPDConfig())),
    ("GMA", GMA_M(GeometricMovingAverageConfig())),
    ("ECDD", ECDDWT_M(ECDDWTConfig())),
    ("EDDM", EDDM_M(EDDMConfig()))

]


def task(stream_path, model, dd):
    stream = CSVStream("{}".format(stream_path))
    stream_name = os.path.splitext(os.path.basename(stream_path))[0]
    stream_output = os.path.dirname(stream_path).replace("datasets", "output")
    print(stream_output)
    model_name, model = model
    dd_name, dd = dd
    if type(dd) == MCADWIN:
        dd = MCADWIN(n_classes=stream.n_classes)
    exp_name = "{}_{}_{}".format(model_name, dd_name, stream_name)
    print("Running {}...".format(exp_name))
    exp = Experiment(exp_name, stream_output, model, dd, stream, stream_size=100000)

    exp.run()

    exp.save()


for model in models:
    PATH = "./datasets/"
    EXT = "*.csv"
    streams = [
        file
        for path, subdir, files in os.walk(PATH)
        for file in glob(os.path.join(path, EXT))
    ]
    import random

    random.shuffle(streams)

    out = Parallel(n_jobs=16)(
        delayed(task)(stream, model, dd)
        for stream, dd in itertools.product(streams, dds)
    )
