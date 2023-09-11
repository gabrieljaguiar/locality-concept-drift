from river.drift.binary import DDM


class DDM_OCI(DDM):
    def __init__(
        self,
        warm_start: int = 30,
        warning_threshold: float = 2,
        drift_threshold: float = 3,
    ):
        super().__init__(warm_start, warning_threshold, drift_threshold)
