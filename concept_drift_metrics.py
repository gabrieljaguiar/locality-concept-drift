import pandas as pd
import os
from glob import glob

# drift_alerts_HT_ADWIN_no_drift_rbf_c_2_f_5_1_1
# HT_RDDM_single_class_local_splitting_cluster_ds_1_c_10_ca_1_f_2_1_1

TP_WINDOW = 5000

classifiers = ["HT"]

dds = [
    "ADWIN",
    "PageHinkley",
    "HDDM",
    "KSWIN",
    "DDM",
    "RDDM",
]

scenarios = [
    #"multi_class_global",
    #"multi_class_local",
    "single_class_global",
    #"single_class_local",
]


metrics = []

for c in classifiers:
    for dd in dds:
        for scenario in scenarios:
                PATH = "./output/"
                EXT = "drift_alerts_{}_{}_{}_*.csv".format(c, dd, scenario)
                print (EXT)
                streams = [
                    file
                    for path, subdir, files in os.walk(PATH)
                    for file in glob(os.path.join(path, EXT))
                ]
                if scenario != "no_drift":
                    for file in streams:
                        splited_name = file.split("_")
                        index_base = splited_name.index(scenario.split("_")[-1])
                        index_ds = splited_name.index("ds") + 1
                        index_ca = splited_name.index("ca") + 1
                        index_c = splited_name.index("c") + 1
                        index_f = splited_name.index("f") + 1
                        drift_speed = int(splited_name[index_ds])
                        class_affected = int(splited_name[index_ca])
                        n_classes = int(splited_name[index_c])
                        n_features = int(splited_name[index_f])
                        difficulty = "_".join(splited_name[(index_base+1):(index_ds-1)])
                        
                        df = pd.read_csv(file)
                        drift_position = 50000 - int(drift_speed) - 1
                        
                        tp = 0
                        fp = 0
                        fn = 0
                        
                        for _, row in df.iterrows():
                            drift_idx = row["idx"]
                            if (drift_idx >= drift_position and drift_idx <= drift_position + TP_WINDOW):
                                if tp == 0:
                                    tp += 1
                                    delay = abs(drift_position - drift_idx)
                                else:
                                    fp += 1
                            else:
                                fp +=1
                            
                            if (drift_idx >= drift_position + TP_WINDOW and tp == 0):
                                fn += 1
                                delay = TP_WINDOW
                        
                        
                        metric = {
                            "classifier": c,
                            "drift_detector": dd,
                            "scenario": scenario,
                            "difficulty": difficulty,
                            "n_classes": n_classes,
                            "n_features": n_features,
                            "drift_speed": drift_speed,
                            "classes_affected": class_affected,
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "delay": delay,
                            #"file": file,
                        }

                        
                        metrics.append(metric)
                    
                    

metric_df = pd.DataFrame(metrics)
metric_df.sort_values(by=["scenario", "difficulty", "n_classes", "n_features", "drift_speed", "classes_affected", "drift_detector"], inplace=True)
metric_df.to_csv("single_class_global_concept_drift_metrics.csv", index=None)
                    