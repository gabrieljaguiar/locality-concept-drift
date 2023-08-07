import numbers
from river.base.drift_detector import DriftDetector
from river.drift import ADWIN
from river.base import DriftDetector
from typing import List


class MCADWIN(DriftDetector):
    def __init__(self, n_classes=2):
        self.n_classes: int = n_classes
        self.drift_detectors: List[DriftDetector] = [ADWIN()] * self.n_classes
        self.classProportions: List[float] = [0] * self.n_classes
        self.theta: float = 0.99
        self.drift_threshold: float = 0.5
        self.local_drift: bool = False

    def update(self, y, error) -> DriftDetector:
        self.local_drift = False
        self._drift_detected = False
        for j in range(0, len(self.classProportions)):
            self.classProportions[j] = self.theta * self.classProportions[j] + (
                1.0 - self.theta
            ) * (1 if y == j else 0)

        self.drift_detectors[y].update(error)
        drift_detection = [0] * self.n_classes
        for j in range(0, self.n_classes):
            if self.drift_detectors[j].drift_detected:
                drift_detection[j] = 1
                self.local_drift = True

        if sum(drift_detection) / len(drift_detection) > self.drift_threshold:
            self._drift_detected = True
