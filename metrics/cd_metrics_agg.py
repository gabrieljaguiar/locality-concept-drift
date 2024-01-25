import pandas as pd
import os
from glob import glob


scenarios = [
    "multi_class_global",
    "multi_class_local",
    "single_class_global",
    "single_class_local",
]

dfs = []

for scenario in scenarios:
    df_specific = pd.read_csv("added_{}_concept_drift_metrics.csv".format(scenario))
    grouped = df_specific.groupby(by=["drift_detector"])
    agg = grouped[["tp", "fn", "fp", "delay"]].sum()
    agg["delay"] = agg["delay"] / grouped["delay"].count()
    filt_group = df_specific.loc[df_specific["tp"] == 1].groupby(by=["drift_detector"])["delay"]
    agg["delay_filt"] = filt_group.sum() / filt_group.count()
    local_agg_1 = agg
    local_agg_1.reset_index().to_csv("added_{}_global_metrics.csv".format(scenario))
    
    grouped = df_specific.groupby(by=["drift_detector", "difficulty"])
    agg = grouped[["tp", "fn", "fp", "delay"]].sum()
    agg["delay"] = agg["delay"] / grouped["delay"].count()
    filt_group = df_specific.loc[df_specific["tp"] == 1].groupby(by=["drift_detector", "difficulty"])["delay"]
    agg["delay_filt"] = filt_group.sum() / filt_group.count()
    local_agg_2 = agg
    local_agg_2.reset_index().to_csv("added_{}_difficulty_metrics.csv".format(scenario))
    
    grouped = df_specific.groupby(by=["drift_detector", "n_classes", "n_features"])
    agg = grouped[["tp", "fn", "fp", "delay"]].sum()
    agg["delay"] = agg["delay"] / grouped["delay"].count()
    filt_group = df_specific.loc[df_specific["tp"] == 1].groupby(by=["drift_detector", "n_classes", "n_features"])["delay"]
    agg["delay_filt"] = filt_group.sum() / filt_group.count()
    local_agg_3 = agg
    local_agg_3.reset_index().to_csv("added_{}_specs_metrics.csv".format(scenario))

    dfs.append(df_specific)

full_df = pd.concat(dfs, axis=0)

grouped = full_df.groupby(by=["drift_detector"])
agg = grouped[["tp", "fn", "fp"]].sum()
agg["delay_full"] = grouped["delay"].sum() / grouped["delay"].count()
filt_group = full_df.loc[full_df["tp"] == 1].groupby(by=["drift_detector"])["delay"]
agg["delay_filt"] = filt_group.sum() / filt_group.count()
global_agg = agg
global_agg.reset_index().to_csv("added_global_metrics.csv")

"""
scenario = "no_drift"
df_specific = pd.read_csv("added_{}_concept_drift_metrics.csv".format(scenario))
grouped = df_specific.groupby(by=["drift_detector"])
agg = grouped[["tp", "fn", "fp", "delay"]].sum()
agg["delay"] = agg["delay"] / grouped["delay"].count()
filt_group = df_specific.loc[df_specific["tp"] == 1].groupby(by=["drift_detector"])["delay"]
agg["delay_filt"] = filt_group.sum() / filt_group.count()
local_agg_1 = agg
local_agg_1.reset_index().to_csv("added_{}_global_metrics.csv".format(scenario))
"""