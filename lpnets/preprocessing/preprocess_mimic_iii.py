
# first run https://github.com/sindhura97/STraTS/blob/main/src/preprocess_mimic_iii_large.py

from dotenv import load_dotenv
load_dotenv()
from pp_utils import *
import os
import pickle
from pathlib import Path

ROOT = Path(os.environ["ICU_PATH"]) / 'MIMIC_III'

IN_PATH = ROOT / "data"

OUT_PATH = ROOT / "saved_data"


path_to_strats = "/home/christel-sirocchi/GitHub/DATASETS/MIMIC_III/preprocessed_strats/mimic_iii.pkl"

with open(path_to_strats, "rb") as f:
    events, oc, train_ids, valid_ids, test_ids = pickle.load(f)

events["ts_id"] = events["ts_id"].astype("int").astype("str")

events.columns = ["ts_id","minute","itemid","value","table"]

events.drop(columns=["table"], inplace=True)

demo = events[events["itemid"].isin(["Age","Gender"])].set_index(["ts_id","itemid"]).drop(columns=["minute"]).reset_index()

demo = demo.pivot(index="ts_id",columns="itemid",values="value")

demo.columns = ["age","gender"]

demo = demo.reset_index()

oc.columns = ["ts_id","hadm_id","subject_id","label"]

oc["ts_id"] = oc["ts_id"].astype("int").astype("str")

oc = oc.merge(demo, on="ts_id", how="left")

oc = oc.drop(columns="hadm_id").rename(columns={"ts_id":"hadm_id"})

events = events[~events["itemid"].isin(["Age", "Gender"])]

events = events[events["minute"]>=0]

events = events.rename(columns={"ts_id":"hadm_id"})

events = events.merge(oc[["hadm_id","subject_id"]], on="hadm_id")

events["hour"] = (events["minute"] / 60).round().astype(int)

events["2hour"] = (events["minute"] / 120).round().astype(int)

events["6hour"] = (events["minute"] / 360).round().astype(int)

events = events[['subject_id', 'hadm_id', 'itemid', 'value', 'minute', 'hour', '2hour', '6hour']]

events.to_csv(OUT_PATH / "mimic_iii_features_to_ts.csv.gz")

oc.to_csv(OUT_PATH / "mimic_iii_cohort.csv.gz")

print("[Checkpoint] Preparing fold splits")

compute_splits_stf(oc, OUT_PATH)