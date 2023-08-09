from river.datasets.base import SyntheticDataset
from random import Random


class HyperplaneMC(SyntheticDataset):
    def __init__(
        self,
        task,
        n_features,
        n_samples=None,
        n_classes=None,
        n_outputs=None,
        sparse=False,
        seed=42,
    ):
        super().__init__(task, n_features, n_samples, n_classes, n_outputs, sparse)
        self._rng = Random(seed)

    def _generate_next_sample(self):
        attVals: list[float] = [0.0] * self.n_features
        sum = 0.0
        sumWeights = 0.0

        for i in range(self.n_features):
            attVals[i] = self._rng()
            sum += self.weights[i] * attVals[i]
            sumWeights += self.weights[i]

        classLabel = 0

        ratio = sum / sumWeights
        sumRoulette = 0.0

        for i in range(self.n_classes):
            sumRoulette += 1.0 / self.n_classes
            if sumRoulette >= ratio:
                classLabel = i
                break
