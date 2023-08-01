from river import datasets
from river.datasets import synth
from typing import Dict
import random
import math


class MultiClassDrift(datasets.base.SyntheticDataset):
    def __init__(self, stream: datasets.base.SyntheticDataset, driftKey: Dict = None):
        self.stream = stream
        self.driftKey = driftKey
        self.n_classes = self.stream.n_classes
        self.streamIterator = iter(self.stream)

        if self.driftKey == None:
            self.driftKey = {}
            for i in range(0, self.n_classes):
                self.driftKey[i] = i

    def __iter__(self):
        while True:
            x, y = next(self.initialStreamIterator)
            yield x, self.driftKey.get(y)
