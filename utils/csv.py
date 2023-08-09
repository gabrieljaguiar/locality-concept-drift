from river.stream import iter_pandas
from river.datasets.base import SyntheticDataset
import pandas as pd


def save_stream(stream: SyntheticDataset, file: str, size: int):
    stream_df_x = []
    stream_df_y = []
    for x, y in stream.take(size):
        stream_df_x.append(x)
        stream_df_y.append(y)

    stream_df_x = pd.DataFrame(stream_df_x)
    stream_df_y = pd.DataFrame(stream_df_y)

    stream_df = pd.concat([stream_df_x, stream_df_y], axis=1, ignore_index=True)
    stream_df.to_csv(file, index=None)
