import numbers
from frouros.detectors.concept_drift.streaming.statistical_process_control.rddm import (
    RDDM,
)
from river.base import DriftDetector
from river.base.drift_detector import DriftDetector


class RDDM_M(DriftDetector):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.rddm = RDDM(*args, **kwargs)
        self._drift_detected = False

    def update(self, x: float) -> DriftDetector:
        self.rddm.update(x)
        self._drift_detected = self.rddm.drift
