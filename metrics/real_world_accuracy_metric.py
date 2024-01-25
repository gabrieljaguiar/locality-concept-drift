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



classifiers = ["HT" , "AHT", "HT_DW"]

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
            EXT = "{}_{}_INSECTS-{}.csv".format(c, dd, scenario)
            stream_file = PATH + EXT
            df = pd.read_csv(stream_file)
            accuracy = df["accuracy"].mean()
            gmean = df["gmean"].mean()
            kappa = df["kappa"].mean()
                        
                        
            metric = {
                            "classifier": c,
                            "scenario": scenario,
                            "drift_detector": dd,
                            "accuracy":accuracy,
                            "kappa": kappa,
                            "gmean": gmean,
                        }

                        
            metrics.append(metric)
                
    i = 0 
    custom_dict = {
        "accuracy": 0,
        "kappa": 1,
        "gmean": 2,
    }
    for dd in dds:
        custom_dict[dd] = i
        i += 1

    metric_df = pd.DataFrame(metrics)
    #print (metrics)
    metric_df.sort_values(by=["classifier", "scenario", "drift_detector"], inplace=True)
    metric_df.to_csv("real_world_predictive_metrics.csv", index=None)
