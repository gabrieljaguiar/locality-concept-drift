from river.datasets.base import SyntheticDataset
from river import datasets
from random import Random


class HyperplaneMC(SyntheticDataset):
    def __init__(
        self,
        n_features,
        n_samples=None,
        n_classes=None,
        n_outputs=None,
        sparse=False,
        seed=42,
    ):
        super().__init__(
            datasets.base.MULTI_CLF, n_features, n_samples, n_classes, n_outputs, sparse
        )
        self._rng = Random(seed)
        self.weights = [self._rng.random() for i in range(self.n_features)]

    def __iter__(self):
        while True:
            x, y = self._generate_next_sample()
            yield x, y

    def _generate_next_sample(self):
        attributes = []
        sum = 0.0
        sumWeights = 0.0

        for i in range(self.n_features):
            attributes.append(self._rng.random())
            sum += self.weights[i] * attributes[i]
            sumWeights += self.weights[i]

        classLabel = 0

        ratio = sum / sumWeights
        sumRoulette = 0.0

        for i in range(self.n_classes):
            sumRoulette += 1.0 / self.n_classes
            if sumRoulette >= ratio:
                classLabel = i
                break

        return attributes, classLabel
