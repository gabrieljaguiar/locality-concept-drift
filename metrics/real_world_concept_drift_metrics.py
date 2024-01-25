import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import numpy as np

# drift_alerts_HT_ADWIN_no_drift_rbf_c_2_f_5_1_1
# HT_RDDM_single_class_local_splitting_cluster_ds_1_c_10_ca_1_f_2_1_1

def find_nearest(array, value):
    if (len(array) == 0):
        return None
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

TP_WINDOW = 5000

classifiers = ["HT"]

dds = [
    "ADWIN",
    "PageHinkley",
    "HDDM",
    "KSWIN",
    "DDM",
    "RDDM",
    "STEPD",
    "ECDD",
    "EDDM",
    "FHDMM",
    "FHDMMS"
]

scenarios = [
    "no_drift",
]


drift_points = {
    "abrupt_balanced_norm": [14.352, 19.500, 33.240, 38.682, 39.510],
    "abrupt_imbalanced_norm": [83.859, 128.651, 182.320, 242.883, 268.380],
    "gradual_balanced_norm": [14.028],
    "gradual_imbalanced_norm": [58.159],
    "incremental-abrupt_balanced_norm": [26.568, 53.364],
    "incremental-abrupt_imbalanced_norm": [150.683, 301.365],
    "incremental-reoccurring_balanced_norm": [26.568, 53.364],
    "incremental-reoccurring_imbalanced_norm": [150.683, 301.365],
}

metrics = []
for scenario in drift_points.keys():

    for c in classifiers:
        for dd in dds:
            PATH = "../real-world-output/"
            EXT = "drift_alerts_HT_{}_INSECTS-{}.csv".format(dd, scenario)
            stream_file = PATH + EXT
            df = pd.read_csv(stream_file)

            #print (df)

            tp = 0
            fp = 0
            fn = 0
            delay = 0
            #print (drift_points)
            
            drift_positions = drift_points.get(scenario).copy()
            for _, row in df.iterrows():
                drift_idx = row["idx"]
                drift_position = find_nearest(drift_positions, drift_idx)
                #if ((drift_position is not None) and (drift_position != drift_positions[0])):
                #    drift_positions.remove(drift_positions[0])
                #    fn += 1
                if (
                            drift_position is not None
                            and drift_idx >= drift_position
                            and drift_idx <= drift_position + TP_WINDOW
                        ):
                    
                    tp += 1
                    delay = delay + abs(drift_position - drift_idx)
                    drift_positions.remove(drift_position)
                else:
                    fp += 1
            
            fn = len (drift_points.get(scenario)) - tp

            metric = {
                        "classifier": c,
                        "scenario": scenario,
                        "drift_detector": dd,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "delay": delay/len(drift_points.get(scenario)),
                    }

            metrics.append(metric)
metric_df = pd.DataFrame(metrics)
metric_df.to_csv("real_world_concept_drift_metrics.csv", index=None)


#metric_df = pd.read_csv("real_world_concept_drift_metrics.csv")
metric_df = metric_df.pivot(
        index=[
            "scenario",
        ],
        columns=["drift_detector"],
        values=["tp", "fp", "fn", "delay"],
    )
custom_dict = {"tp": 0, "fp": 1, "fn": 2, "delay": 3}
i = 0
for dd in dds:
    custom_dict[dd] = i
    i += 1
metric_df = metric_df.swaplevel(0, 1, axis=1).sort_index(
        axis=1, level=[0, 1], key=lambda x: x.map(custom_dict)
    )
metric_df.reset_index(inplace=True)
print (metric_df)
metric_df.to_csv(
        "real_world_pivoted_concept_drift_metrics.csv", index=None
    )
"""               
    i = 0
    


    metric_df = pd.DataFrame(metrics)
    print(metrics)
    metric_df.sort_values(
        by=[
            "scenario",
            "difficulty",
            "n_classes",
            "n_features",
            "drift_speed",
            "classes_affected",
            "drift_detector",
        ],
        inplace=True,
    )
    metric_df.to_csv("{}_concept_drift_metrics.csv".format(scenario), index=None)
    metric_df = metric_df.pivot(
        index=[
            "scenario",
            "difficulty",
            "n_classes",
            "n_features",
            "drift_speed",
            "classes_affected",
        ],
        columns=["drift_detector"],
        values=["tp", "fp", "fn", "delay"],
    )
    metric_df = metric_df.swaplevel(0, 1, axis=1).sort_index(
        axis=1, level=[0, 1], key=lambda x: x.map(custom_dict)
    )

"""