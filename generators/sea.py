import random
from river import datasets


class SEAMod(datasets.base.SyntheticDataset):
    def __init__(self, variant=0, noise=0.0, seed: int = None):
        super().__init__(n_features=3, task=datasets.base.BINARY_CLF)

        if variant not in (0, 1, 2, 3, 4, 5):
            raise ValueError("Unknown variant, possible choices are: 0, 1, 2, 3")

        self.variant = variant
        self.noise = noise
        self.seed = seed
        self._threshold = {0: 10, 1: 2.5, 2: 17.5, 3: 8.3, 4: 14.5, 5: 5.6}[variant]

    def __iter__(self):
        rng = random.Random(self.seed)

        while True:
            x = {i: rng.uniform(0, 10) for i in range(3)}
            y = x[0] + x[1] > self._threshold

            if self.noise and rng.random() < self.noise:
                y = not y

            yield x, y

    @property
    def _repr_content(self):
        return {**super()._repr_content, "Variant": str(self.variant)}
