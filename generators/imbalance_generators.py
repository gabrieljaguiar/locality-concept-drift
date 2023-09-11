import random
from river import datasets


def getClassRatios(n_classes: int, imbalance: bool = True):
    if not imbalance:
        return [1 / n_classes for i in range(n_classes)]
    else:
        proportions = [1] * n_classes
        proportions[len(proportions) - 1] = 1 / n_classes
        return [proportions[i] / sum(proportions) for i in range(0, n_classes)]


class BinaryImbalancedStream(datasets.base.SyntheticDataset):
    def __init__(
        self,
        generator: datasets.base.SyntheticDataset,
        imbalanceRatio: float,
        seed: int = 42,
    ):
        self.generator = generator
        self.imbalanceRatio = imbalanceRatio
        self.seed = seed
        self._rng = random.Random(self.seed)

        # assert (self.generator.n_classes == 2, "Binary generators only")

        super().__init__(
            self.generator.task,
            self.generator.n_features,
            self.generator.n_samples,
            self.generator.n_classes,
            self.generator.n_outputs,
        )

    def __iter__(self):
        self.generatorIterator = iter(self.generator)
        while True:
            expectedClass = 1 if self._rng.random() < self.imbalanceRatio else 0
            x, y = next(self.generatorIterator)
            while y != expectedClass:
                # print("searching --- {}".format(self.imbalanceRatio))
                x, y = next(self.generatorIterator)
            yield x, y

    def __repr__(self):
        return self.generator.__repr__()

    def getImbalance(self):
        return self.imbalanceRatio


class MultiClassImbalancedStream(datasets.base.SyntheticDataset):
    def __init__(
        self,
        generator: datasets.base.SyntheticDataset,
        imbalanceRatio: list,
        seed: int = 42,
    ):
        self.generator = generator
        self.imbalanceRatio = imbalanceRatio
        self.seed = seed
        self._rng = random.Random(self.seed)
        self.n_classes = self.generator.n_classes

        assert (
            round(sum(self.imbalanceRatio), 5) == 1
        ), "Sum of probabilities must be 1 - {}".format(
            round(sum(self.imbalanceRatio), 5)
        )
        assert self.n_classes == len(
            self.imbalanceRatio
        ), "Generator number of classes and probability list should have the same size"

        super().__init__(
            self.generator.task,
            self.generator.n_features,
            self.generator.n_samples,
            self.generator.n_classes,
            self.generator.n_outputs,
        )

    def __iter__(self):
        self.generatorIterator = iter(self.generator)
        while True:
            nextClassProbability = self._rng.random()
            classIndex = -1

            while nextClassProbability > 0:
                classIndex += 1
                nextClassProbability -= self.imbalanceRatio[classIndex]
            expectedClass = classIndex
            x, y = next(self.generatorIterator)
            while y != expectedClass:
                x, y = next(self.generatorIterator)
            yield x, y
