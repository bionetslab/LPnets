from pathlib import Path
import os

PATIENTS_NETWORKS = Path(__file__).resolve().parents[2]
LPNETS = Path(__file__).resolve().parents[1]
GRAPH_RESULTS = PATIENTS_NETWORKS / "graph_results"

RANDOM_SEED = 42

BASE_MIMIC_CHEMO_PATH = Path(os.environ["MIMIC_CHEMO_PATH"])
BASE_UKER_CHEMO_PATH = Path(os.environ["UKER_CHEMO_PATH"])
BASE_ICU_PATH = Path(os.environ["ICU_PATH"])
BASE_OMIC_PATH = Path(os.environ["OMIC_PATH"])

DATASET_CONFIG = {
    "physionet_2012_icu": {
        "base": BASE_ICU_PATH / "Physionet_2012" / "saved_data",
        "data_file": "physionet_2012_features_to_ts.csv.gz",
        "cohort_file": "physionet_2012_cohort.csv.gz",
        "CV_folds_path": ".",
    },
    "physionet_2019_icu": {
        "base": BASE_ICU_PATH / "Physionet_2019" / "saved_data",
        "data_file": "physionet_2019_features_to_ts.csv.gz",
        "cohort_file": "physionet_2019_cohort.csv.gz",
        "CV_folds_path": ".",
    },
    "mimic_iii_icu": {
        "base": BASE_ICU_PATH / "MIMIC_III" / "saved_data",
        "data_file": "mimic_iii_features_to_ts.csv.gz",
        "cohort_file": "mimic_iii_cohort.csv.gz",
        "CV_folds_path": ".",
    }
}

omic_pancan_datasets = ["lung_tumours", "brain_tumours", "colorectal_tumours", "kidney_tumours", "gi_adenocarcinomas", "gyn_adenocarcinomas", "melanomas", "adrenal_tumours"]

omic_pancan_datasets_1000 = [ds + "1000" for ds in omic_pancan_datasets]
omic_pancan_datasets_2000 = [ds + "2000" for ds in omic_pancan_datasets]
omic_pancan_datasets_HVG_1000 = [ds + "_HVG_1000" for ds in omic_pancan_datasets]

omic_pancan_datasets_all = omic_pancan_datasets_1000 + omic_pancan_datasets_2000 + omic_pancan_datasets_HVG_1000

for omic in omic_pancan_datasets_all:
    DATASET_CONFIG[omic] = {
        "base": BASE_OMIC_PATH / omic / "saved_data",
        "data_file": f"{omic}_features_to_ts.csv.gz",
        "cohort_file": f"{omic}_cohort.csv.gz",
        "CV_folds_path": ".",        
    }

mimic_datasets = ["mimic_cohort_NF_30_days", "mimic_cohort_aplasia_45_days"]

for cohort in mimic_datasets:
    DATASET_CONFIG[cohort] = {
        "base": BASE_MIMIC_CHEMO_PATH,
        "data_file": Path("processed_admission_features_for_ts") / cohort / f"{cohort}_admissions_labs_14_days_to_ts.csv.gz",
        "cohort_file": Path("cohorts") / f"{cohort}.csv.gz",
        "CV_folds_path": Path("folds") / cohort,
        "top_features": Path("top_features") / "mimic_top100_features.pkl"
    }

uker_datasets = ["uker_cohort_NF_30_days", "uker_cohort_aplasia_45_days"]

for cohort in uker_datasets:
    DATASET_CONFIG[cohort] = {
        "base": BASE_UKER_CHEMO_PATH,
        "data_file": Path("processed_admission_features_for_ts") / cohort / f"{cohort}_admissions_labs_14_days_to_ts.csv.gz",
        "cohort_file": Path("cohorts") / f"{cohort}.csv.gz",
        "CV_folds_path": Path("folds") / cohort
    }
