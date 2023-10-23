import numbers
from frouros.detectors.concept_drift.streaming.statistical_process_control.rddm import (
    RDDM,
)
from frouros.detectors.concept_drift.streaming.window_based.stepd import STEPD
from frouros.detectors.concept_drift.streaming.change_detection.geometric_moving_average import GeometricMovingAverage
from frouros.detectors.concept_drift.streaming.statistical_process_control.ecdd import ECDDWT
from frouros.detectors.concept_drift.streaming.statistical_process_control.eddm import EDDM
from river.base import DriftDetector
from river.base.drift_detector import DriftDetector, BinaryDriftAndWarningDetector


class RDDM_M(BinaryDriftAndWarningDetector):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.rddm = RDDM(*args, **kwargs)
        self._drift_detected = False

    def update(self, x: float) -> DriftDetector:
        self.rddm.update(x)
        self._drift_detected = self.rddm.drift
        self._warning_detected = self.rddm.warning


class STEPD_M(BinaryDriftAndWarningDetector):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.step_d = STEPD(*args, **kwargs)
        self._drift_detected = False

    def update(self, x: float) -> DriftDetector:
        self.step_d.update(x)
        self._drift_detected = self.step_d.drift
        self._warning_detected = self.step_d.warning

class EDDM_M(BinaryDriftAndWarningDetector):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.eddm = EDDM(*args, **kwargs)
        self._drift_detected = False

    def update(self, x: float) -> DriftDetector:
        self.eddm.update(x)
        self._drift_detected = self.eddm.drift
        self._warning_detected = self.eddm.warning


class ECDDWT_M(BinaryDriftAndWarningDetector):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ecddwt = ECDDWT(*args, **kwargs)
        self._drift_detected = False

    def update(self, x: float) -> DriftDetector:
        self.ecddwt.update(x)
        self._drift_detected = self.ecddwt.drift
        self._warning_detected = self.ecddwt.warning
        
class GMA_M(DriftDetector):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gma = GeometricMovingAverage(*args, **kwargs)
        self._drift_detected = False

    def update(self, x: float) -> DriftDetector:
        self.gma.update(x)
        self._drift_detected = self.gma.drift