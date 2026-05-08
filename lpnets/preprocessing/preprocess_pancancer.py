# https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
#cohort: TCGA Pan-Cancer (PANCAN)
# download gene expression data at https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz
# download target at https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import pandas as pd
import numpy as np
import os
from pp_utils import *

ROOT = Path(os.environ["OMIC_PATH"])

IN_PATH = ROOT / "raw_data"

# gene expression data 
ge_data = pd.read_csv(IN_PATH / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz", sep="\t", index_col=0).T.rename_axis("hadm_id")

# demographic data
ge_demo = pd.read_csv(IN_PATH / "Survival_SupplementalTable_S1_20171025_xena_sp", sep="\t").set_index("sample")[["gender","age_at_initial_pathologic_diagnosis"]].rename_axis("hadm_id")

# target data
ge_target = pd.read_csv(IN_PATH / "TCGA_phenotype_denseDataOnlyDownload.tsv.gz", sep="\t").set_index("sample").rename_axis("hadm_id")

# preprocess demographic data
ge_demo.rename(columns={"age_at_initial_pathologic_diagnosis": "age"}, inplace=True)
ge_demo["gender"] = ge_demo["gender"].str.upper().map({"FEMALE": "F", "MALE": "M"})
ge_target = ge_target.join(ge_demo)

# consider only primary tumor targets
ge_target_pt = ge_target[ge_target["sample_type"] == "Primary Tumor"]

# select relevant ids
comidx = ge_data.index.intersection(ge_target_pt.index)
ge_target_pt = ge_target_pt.loc[comidx]

# extracted classification tasks
cohort_dict = {

    # Lung subtype
    "lung_tumours" : [
        "lung adenocarcinoma",
        "lung squamous cell carcinoma"
    ],

    # Brain tumours
    "brain_tumours" : [
        "glioblastoma multiforme",
        "brain lower grade glioma"
    ],

    # Colorectal tumours
    "colorectal_tumours" : [
        "colon adenocarcinoma",
        "rectum adenocarcinoma"
    ],

    # Kidney subtypes
    "kidney_tumours" : [
        "kidney clear cell carcinoma",
        "kidney papillary cell carcinoma",
        "kidney chromophobe"
    ],
    
    # gastrointestinal adenocarcinomas
    "gi_adenocarcinomas" : [
        "colon adenocarcinoma",
        "stomach adenocarcinoma",
        "pancreatic adenocarcinoma"
    ],

    # gynaecological adenocarcinomas
    "gyn_adenocarcinomas" : [
        "ovarian serous cystadenocarcinoma",
        "uterine corpus endometrioid carcinoma",
        "cervical & endocervical cancer"
    ],

    #melanomas
    "melanomas": [
        "skin cutaneous melanoma",
        "uveal melanoma"
    ],

    #adrenal tumours
    "adrenal_tumours":[
     "adrenocortical cancer",
     "pheochromocytoma & paraganglioma"  
    ]
}

for topN in [1000,2000]:
    
    top_N_genes = ge_data.var(axis=0, skipna=True).nlargest(topN).index
    top_ge_data = ge_data.loc[comidx, top_N_genes]
    
    for cohort_disease, subtypes in cohort_dict.items():

        cohort = cohort_disease + str(topN)
        
        print(f"\n\n[Checkpoint] Preparing data for {cohort} cohort")
    
        label_map = {cls: i for i, cls in enumerate(subtypes)}
    
        OUT_PATH = ROOT / cohort / "saved_data"
        OUT_PATH.mkdir(parents=True, exist_ok=True)
    
        subidx = ge_target_pt[ge_target_pt["_primary_disease"].isin(subtypes)].index
    
        # process target
        print(f"[Checkpoint] {cohort} - preparing cohort data")
    
        ge_target_cohort = ge_target_pt.loc[subidx]
    
        ge_target_cohort["label"] = ge_target_cohort["_primary_disease"].map(label_map)
    
        ge_target_cohort = ge_target_cohort.reset_index()
    
        ge_target_cohort.to_csv(OUT_PATH / f"{cohort}_cohort.csv.gz")
    
        #print(ge_target_cohort.head())
      
        # process data
        print(f"[Checkpoint] {cohort} - preparing omic data")
    
        ge_data_cohort = top_ge_data.loc[subidx]
    
        ge_top_long = pd.melt(ge_data_cohort.fillna(0).reset_index(), id_vars=["hadm_id"], var_name="itemid", value_name="value")
    
        ge_top_long.to_csv(OUT_PATH / f"{cohort}_features_to_ts.csv.gz")
    
        #print(ge_top_long.head())
    
        # compute folds
        print(f"[Checkpoint] {cohort} - preparing folds")
    
        compute_splits_stf(ge_target_cohort, OUT_PATH)