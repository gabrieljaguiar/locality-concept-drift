from glob import glob
import os
import pandas as pd

PATH = "./datasets/"
EXT = "*.csv"
streams = [
    file
    for path, subdir, files in os.walk(PATH)
    for file in glob(os.path.join(path, EXT))
]

for stream in streams:
    df = pd.read_csv(stream)
    stream_name = os.path.splitext(os.path.basename(stream))[0]
    df.iloc[:, :-1] = df.iloc[:, :-1].round(4)
    df.to_csv("./resized_datasets/{}.csv".format(stream_name), index=None)
