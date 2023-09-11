from river.stream import iter_arff
from scipy.io.arff import arffread
from river import base
from river.stream import utils as ru
from river.datasets import base as dsbase


class ArffStream(dsbase.Dataset):
    def __init__(self, filepath, compression="infer"):
        self.buffer = filepath
        if not hasattr(self.buffer, "read"):
            self.buffer = ru.open_filepath(self.buffer, compression)

        try:
            rel, attrs = arffread.read_header(self.buffer)
        except ValueError as e:
            msg = f"Error while parsing header, error was: {e}"
            raise arffread.ParseArffError(msg)

        self.features = [attr.name for attr in attrs[:-1]]

        self.target = [attr.name for attr in attrs[-1:]]
        self.target_range = [attr.range for attr in attrs[-1:]][0]

        self.feat_types = [
            float if isinstance(attr, arffread.NumericAttribute) else None
            for attr in attrs
        ]

        self.n_classes = len(self.target_range)
        self.n_features = len(self.features)
        self.name = rel

    def __iter__(self):
        for r in self.buffer:
            if len(r) <= 1:
                continue

            if len(r.rstrip().split(",")) < len(self.feat_types):
                continue

            r = r.replace("?", "0")
            # line = r.rstrip().split(",")
            x = {
                name: typ(val) if typ else val
                for name, typ, val in zip(
                    self.features + self.target, self.feat_types, r.rstrip().split(",")
                )
            }
            try:
                y = x.pop(self.target[0]) if self.target else None
                if isinstance(y, str):
                    # print(y)
                    y = y.replace('"', "")
                    y = y.replace("'", "")
                    # print(y)
            except KeyError as e:
                print(r)
                raise e

            yield x, y
