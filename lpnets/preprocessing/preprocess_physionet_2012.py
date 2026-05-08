from dotenv import load_dotenv
load_dotenv()

#https://physionet.org/content/challenge-2012/1.0.0/#files-panel

from tqdm import tqdm
import os
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from pp_utils import *

ROOT = Path(os.environ["ICU_PATH"]) / 'Physionet_2012'

IN_PATH = ROOT / "data"

OUT_PATH = ROOT / "saved_data"

def read_ts(raw_data_path, set_name):
    ts = []
    pbar = tqdm(os.listdir(raw_data_path+'/set-'+set_name), 
                desc='Reading time series set '+set_name)
    for f in pbar:
        data = pd.read_csv(raw_data_path+'/set-'+set_name+'/'+f).iloc[1:]
        data = data.loc[data.Parameter.notna()]
        if len(data)<=5:
            continue
        data = data.loc[data.Value>=0] # neg Value indicates missingness.
        data['RecordID'] = f[:-4]
        ts.append(data)
    ts = pd.concat(ts)
    ts.Time = ts.Time.apply(lambda x:int(x[:2])*60
                            +int(x[3:])) # No. of minutes since admission.
    ts.rename(columns={'Time':'minute', 'Parameter':'variable', 
                       'Value':'value', 'RecordID':'ts_id'}, inplace=True)
    return ts


def read_outcomes(raw_data_path, set_name):
    oc = pd.read_csv(raw_data_path+'/Outcomes-'+set_name+'.txt', 
                     usecols=['RecordID', 'Length_of_stay', 'In-hospital_death'])
    oc['subset'] = set_name
    oc.RecordID = oc.RecordID.astype(str)
    oc.rename(columns={'RecordID':'ts_id', 'Length_of_stay':'length_of_stay', 
                       'In-hospital_death':'in_hospital_mortality'}, inplace=True)
    return oc


ts = pd.concat([read_ts(str(IN_PATH), set_name) for set_name in ['a','b','c']])
oc = pd.concat([read_outcomes(str(IN_PATH), set_name) for set_name in ['a','b','c']])
ts_ids = sorted(list(ts.ts_id.unique()))
oc = oc.loc[oc.ts_id.isin(ts_ids)]
ts = ts.drop_duplicates()

# split vars

ts_vars = ['Weight', 'GCS', 'HR', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Temp', 'FiO2', 'MechVent', 'Urine', 'DiasABP', 'MAP', 'SysABP', 
           'pH', 'PaCO2', 'PaO2', 'SaO2','Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Creatinine', 'Glucose', 'HCO3', 'HCT', 
           'Mg', 'Platelets', 'K', 'Na', 'WBC', 'Lactate',  'RespRate', 'Height', 'TroponinT', 'TroponinI', 'Cholesterol']

demo_vars = ['Age', 'Gender']

excluded = ['ICUType']

# preprocess temporal data

print("[Checkpoint] Preparing time series data")

features = ts[ts["variable"].isin(ts_vars)].reset_index(drop=True)
features.columns = ['minute', 'itemid', 'value', 'hadm_id']
features["hour"] = (features["minute"] / 60).round().astype(int)
features["2hour"] = (features["minute"] / 120).round().astype(int)
features["6hour"] = (features["minute"] / 360).round().astype(int)
features["hadm_id"] = features["hadm_id"].astype("str")
features = features[['hadm_id', 'itemid', 'value', 'minute', 'hour', '2hour', '6hour']]

features.to_csv(OUT_PATH / "physionet_2012_features_to_ts.csv.gz")

# preprocess cohort data

print("[Checkpoint] Preparing cohort data")

oc.columns = ['hadm_id', 'length_of_stay', 'label', 'subset']
demo = ts[ts["variable"].isin(demo_vars)].reset_index(drop=True).drop(columns="minute")
demo = demo.set_index(["ts_id","variable"]).unstack("variable")
demo.columns = ["age","gender"]
demo = demo.reset_index(names="hadm_id")
cohort = oc.merge(demo, on="hadm_id")
cohort["hadm_id"] = cohort["hadm_id"].astype("str")
cohort["gender"] = cohort["gender"].map({0:"F",1:"M"})

cohort.to_csv(OUT_PATH / "physionet_2012_cohort.csv.gz")

# compute splits 

print("[Checkpoint] Preparing fold splits")

compute_splits_stf(cohort, OUT_PATH)

"""
for i, f in enumerate(['a', 'b', 'c']):

    # TODO set all seeds to 42 globally
    # train + validation pool (everything except subset f)
    train_val_ids = cohort.loc[cohort.subset != f, 'hadm_id'].astype(str).values
    np.random.seed(123)
    np.random.shuffle(train_val_ids)

    # 80/20 split, test set = subset f
    bp = int(0.8 * len(train_val_ids))
    train_ids = train_val_ids[:bp].astype(str)
    val_ids   = train_val_ids[bp:].astype(str)
    test_ids = cohort.loc[cohort.subset == f, 'hadm_id'].astype(str).values

    # reorder based based on cohort file (for debugging) #TODO remove these lines
    train_ids = cohort.loc[cohort['hadm_id'].astype(str).isin(train_ids), 'hadm_id'].astype(str).values
    val_ids = cohort.loc[cohort['hadm_id'].astype(str).isin(val_ids), 'hadm_id'].astype(str).values

    print(f"\nFOLD {i}")
    print_split_stats(cohort, {"Train": train_ids, "Val": val_ids, "Test":  test_ids})

    with open(os.path.join(OUT_PATH, f'fold_{i}.pkl'), 'wb') as f:
        pickle.dump([train_ids, val_ids, test_ids], f)
"""
