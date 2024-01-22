import numbers
from river import  drift,base
from river.base.drift_detector import DriftDetector
from .fhddm import FHDDM


class ADWINDW(base.DriftAndWarningDetector):
    def __init__(self):
        self.dd = drift.ADWIN()
        self.warning = drift.ADWIN(delta=0.01)
        
    def update(self, x) -> DriftDetector:
        self.dd.update(x)
        self.warning.update(x)
        
        self._drift_detected = self.dd._drift_detected
        self._warning_detected = self.warning._drift_detected        
        
class KSWINDW(base.DriftAndWarningDetector):
    def __init__(self):
        self.dd = drift.KSWIN()
        self.warning = drift.KSWIN(alpha = 0.0075)
        
    def update(self, x) -> DriftDetector:
        self.dd.update(x)
        self.warning.update(x)
        
        self._drift_detected = self.dd._drift_detected
        self._warning_detected = self.warning._drift_detected
    
class PHDW(base.DriftAndWarningDetector):
    def __init__(self):
        self.dd = drift.PageHinkley()
        self.warning = drift.PageHinkley(threshold=100)
        
    def update(self, x) -> DriftDetector:
        self.dd.update(x)
        self.warning.update(x)
        
        self._drift_detected = self.dd._drift_detected
        self._warning_detected = self.warning._drift_detected 
        

class FHDDMDW(base.DriftAndWarningDetector):
    def __init__(self):
        self.dd = FHDDM()
        self.warning = FHDDM(confidence_level = 0.00001)
    
    def update(self, x) -> DriftDetector:
        self.dd.update(x)
        self.warning.update(x)
        
        self._drift_detected = self.dd._drift_detected
        self._warning_detected = self.warning._drift_detected 
        
class FHDDMSDW(base.DriftAndWarningDetector):
    def __init__(self):
        self.dd = FHDDM(short_window_size=20)
        self.warning = FHDDM(confidence_level = 0.00001, short_window_size=20)
        
    def update(self, x) -> DriftDetector:
        self.dd.update(x)
        self.warning.update(x)
        
        self._drift_detected = self.dd._drift_detected
        self._warning_detected = self.warning._drift_detected 