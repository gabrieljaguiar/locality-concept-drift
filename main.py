from river import tree, drift
from experiment import Experiment
from joblib import Parallel, delayed
import itertools
from drift_detectors import RDDM_M

from drift_detectors import (
    RDDM_M,
    EDDM_M,
    STEPD_M,
    ECDDWT_M,
    ADWINDW,
    KSWINDW,
    PHDW,
)
from drift_detectors import ECDDWTConfig, EDDMConfig, RDDMConfig, STEPDConfig
from drift_detectors import MCADWIN
from glob import glob
import os
from utils.csv import CSVStream

models = [
    ("HT", tree.HoeffdingTreeClassifier()),
    ("AHT", tree.HoeffdingAdaptiveTreeClassifier()),
    ("HTDD", drift.DriftRetrainingClassifier(model=tree.HoeffdingTreeClassifier()))
]


dds = [
    ("ADWIN", ADWINDW()),
    ("PageHinkley", PHDW()),
    ("HDDM", drift.binary.HDDM_W()),
    ("KSWIN", KSWINDW()),
    ("DDM", drift.binary.DDM()),
    ("RDDM", RDDM_M(RDDMConfig())),
    ("STEPD", STEPD_M(STEPDConfig())),
    ("ECDD", ECDDWT_M(ECDDWTConfig())),
    ("EDDM", EDDM_M(EDDMConfig())),
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
    if type(model) == drift.DriftRetrainingClassifier:
        model.drift_detector = dd.clone()
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

    out = Parallel(n_jobs=1)(
        delayed(task)(stream, model, dd)
        for stream, dd in itertools.product(streams, dds)
    )
