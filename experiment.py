from river.datasets.base import SyntheticDataset
from river.base import DriftDetector, Classifier
from evaluators.multi_class_evaluator import MultiClassEvaluator
import pandas as pd
from tqdm import tqdm
from drift_detectors import DDM_OCI

# add class imbalance monitoring


class Experiment:
    def __init__(
        self,
        name: str,
        savePath: str,
        model: Classifier,
        driftDetector: DriftDetector,
        stream: SyntheticDataset,
        evaluationWindow: int = 500,
        theta: float = 0.99,
    ) -> None:
        self.name = name
        self.savePath = savePath
        self.model = model
        self.driftDetctor = driftDetector
        self.stream = stream
        self.size = self.stream.n_samples
        self.evaluator = MultiClassEvaluator(evaluationWindow, self.stream.n_classes)
        self.evaluationWindow = evaluationWindow
        self.gracePeriod = 200
        self.theta = theta
        self.classProportions: list = [0] * self.stream.n_classes

    def updateDriftDetector(self, y, y_hat):  # DDM
        x = 1 if (y == y_hat) else 0
        if (type(self.driftDetctor) == DDM_OCI) and (
            y == self.classProportions.index(min(self.classProportions))
        ):
            self.driftDetctor.update(x)
        else:
            self.driftDetctor.update(x)

    def run(self):
        self.metrics = []
        drift_detected = 0
        for i, (x, y) in enumerate(self.stream.take(self.size)):
            if i > self.gracePeriod:
                self.updateDriftDetector(y, self.model.predict_one(x))
                self.evaluator.addResult((x, y), self.model.predict_proba_one(x))
                if self.driftDetctor.drift_detected:
                    drift_detected += 1

                if (i + 1) % self.evaluationWindow == 0:
                    metric = {
                        "idx": i + 1,
                        "accuracy": self.evaluator.getAccuracy(),
                        "gmean": self.evaluator.getGMean(),
                    }

                    for c in range(0, self.stream.n_classes):
                        metric["class_{}".format(c)] = self.evaluator.getClassRecall(c)
                        metric["class_prop_{}".format(c)] = self.classProportions[c]

                    metric["drifts_alerts"] = drift_detected

                    self.metrics.append(metric)

                    drift_detected = 0
            for j in range(0, len(self.classProportions)):
                self.classProportions[j] = self.theta * self.classProportions[j] + (
                    1.0 - self.theta
                ) * (1 if y == j else 0)

            self.model.learn_one(x, y)

    def save(self):
        pd.DataFrame(self.metrics).to_csv("{}/{}.csv".format(self.savePath, self.name))
