import pandas as pd
import os
from glob import glob
from tqdm import tqdm
# drift_alerts_HT_ADWIN_no_drift_rbf_c_2_f_5_1_1
# HT_RDDM_single_class_local_splitting_cluster_ds_1_c_10_ca_1_f_2_1_1

TP_WINDOW = 5000

classifiers = ["AHT", "HT", "HTDD"]

dds = [
    "ADWIN",
    #"PageHinkley",
    #"HDDM",
    #"KSWIN",
    #"DDM",
    #"RDDM",
    #"STEPD",
    #"GMA",
    #"ECDD",
    #"EDDM"
]

scenarios = [
    "multi_class_global",
    "multi_class_local",
    "single_class_global",
    "single_class_local",
    #"no_drift",
]



for scenario in scenarios:
    metrics = []
    for c in classifiers:
        for dd in dds:
                PATH = "../output/"
                EXT = "{}_{}_{}_*.csv".format(c, dd, scenario)
                print (EXT)
                streams = [
                    file
                    for path, subdir, files in os.walk(PATH)
                    for file in glob(os.path.join(path, EXT))
                ]
                #print (streams)
                print (scenario)
                print (dd)
                print (c)
                #print (streams)
                
                for idx, file in tqdm(enumerate(streams), total=len(streams)):
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
                    accuracy = df["accuracy"].mean()
                    gmean = df["gmean"].mean()
                    kappa = df["kappa"].mean()
                        
                        
                    metric = {
                            "classifier": c,
                            "scenario": scenario,
                            "difficulty": difficulty,
                            "n_classes": n_classes,
                            "n_features": n_features,
                            "drift_speed": drift_speed,
                            "classes_affected": class_affected,
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
    metric_df.sort_values(by=["classifier", "scenario",  "difficulty", "n_classes", "n_features", "drift_speed", "classes_affected", "drift_detector"], inplace=True)
    metric_df.to_csv("{}_predictive_metrics.csv".format(scenario), index=None)
    #metric_df = metric_df.pivot(index=["scenario", "difficulty", "n_classes", "n_features", "drift_speed", "classes_affected"], columns=["drift_detector"], values=["tp", "fp", "fn", "delay"])
    #metric_df = metric_df.swaplevel(0, 1, axis=1).sort_index(axis=1, level=[0, 1], key=lambda x: x.map(custom_dict))
    #metric_df.reset_index(inplace=True)
    #metric_df.to_csv("{}_pivoted_concept_drift_metrics.csv".format(scenario), index=None)
                    