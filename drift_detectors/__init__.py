from .frouros import RDDM_M, EDDM_M, STEPD_M, ECDDWT_M, GMA_M
from frouros.detectors.concept_drift.streaming.statistical_process_control.rddm import (
    RDDMConfig,
)

from frouros.detectors.concept_drift.streaming.window_based.stepd import STEPDConfig
from frouros.detectors.concept_drift.streaming.change_detection.geometric_moving_average import GeometricMovingAverageConfig
from frouros.detectors.concept_drift.streaming.statistical_process_control.ecdd import ECDDWTConfig
from frouros.detectors.concept_drift.streaming.statistical_process_control.eddm import EDDMConfig
from .dwd_detectors import ADWINDW, KSWINDW, PHDW, FHDDMSDW, FHDDMDW