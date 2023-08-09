from river.datasets.base import SyntheticDataset


class HyperplaneMC(SyntheticDataset):
    def __init__(
        self,
        task,
        n_features,
        n_samples=None,
        n_classes=None,
        n_outputs=None,
        sparse=False,
    ):
        super().__init__(task, n_features, n_samples, n_classes, n_outputs, sparse)
