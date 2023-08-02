from river.datasets.base import SyntheticDataset
from river.base import DriftDetector, Classifier
from evaluators.multi_class_evaluator import MultiClassEvaluator
from river.drift import ADWIN

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

    def updateDriftDetector(self, y, y_hat):  # DDM
        x = 1 if (y == y_hat) else 0
        self.driftDetctor.update(x)

    def run(self):
        self.metrics = []
        drift_detected = 0
        for i, (x, y) in enumerate(self.stream.take(self.size)):
            if i > self.gracePeriod:
                self.updateDriftDetector(y, self.model.predict_one(x))
                if self.driftDetctor.drift_detected:
                    drift_detected += 1

                if (i + 1) % self.evaluationWindow == 0:
                    print(i + 1)
                    metric = {"idx": i + 1, "G-Mean": self.evaluator.getGMean()}

                    for c in range(0, self.stream.n_classes):
                        metric["class-{}".format(c)] = self.evaluator.getClassRecall(c)

                    metric["drifts_alerts"] = drift_detected

                    self.metrics.append(metric)

                    drift_detected = 0

            self.model.learn_one(x, y)

    def saveExp():
        pass
