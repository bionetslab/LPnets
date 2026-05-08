import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    recall_score, precision_score, balanced_accuracy_score, f1_score
)
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from lpnets.config.constants import GRAPH_RESULTS, DATASET_CONFIG


### LOADING UTILS ###

import pandas as pd
from pathlib import Path
from collections import defaultdict

import pandas as pd
from pathlib import Path
from collections import defaultdict

def load_all_data(out_path, og_path):
    """
    Load metrics for all strategies and feature types.

    Returns:
        metrics[split] = concatenated dataframe (all feature types)
        feature_names[feature_type] = list of columns (for traceability)
    """

    base = Path(out_path)

    metrics = defaultdict(list)
    feature_names = defaultdict(list)

    feature_types = ["node","edge","graph"]
    feature_folders = [f"{f}_features" for f in feature_types]

    # IMPORT METRICS

    for folder in base.iterdir():

        if not folder.is_dir():
            continue

        folder_name = folder.name

        # TS0 (no subfolder)
        if folder_name in feature_folders:
            feature_dir = folder

            for split_file in feature_dir.glob("*_graph_metrics.pkl"):

                split_name = split_file.name.replace("_graph_metrics.pkl", "").rstrip("_")
                df = pd.read_pickle(split_file)
                feature_names[folder_name].extend(df.columns)
                metrics[split_name].append(df)

        # TS1 / TS2 / TS3 (subfolders)
        else:
            for feature_dir in folder.iterdir():

                if not feature_dir.is_dir() or not feature_dir.name.endswith("_features"):
                    continue

                feature_type = feature_dir.name.replace("_features", "")

                for split_file in feature_dir.glob("*_graph_metrics.pkl"):
                    split_name = split_file.name.replace("_graph_metrics.pkl", "").rstrip("_")
                    df = pd.read_pickle(split_file)
                    df.columns = [f"{folder.name}_{c}" for c in df.columns]
                    feature_names[feature_type].extend(df.columns)
                    metrics[split_name].append(df)

    # IMPORT ORIGINAL DATA

    for split_file in og_path.glob("*.pkl"):
        split_name = split_file.name.replace("_data.pkl", "").rstrip("_")
        df = pd.read_pickle(split_file)
        df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) and len(col) == 2 else col for col in df.columns]
        metrics[split_name].append(df)
        feature_names["original"].extend(df.columns)

    # concatenate horizontally per split
    result = {
        split: pd.concat(dfs, axis=1) if dfs else pd.DataFrame()
        for split, dfs in metrics.items()
    }

    # unique feature names per type
    feature_names = {
        k: sorted(set(v))
        for k, v in feature_names.items()
    }

    feature_names["all"] = sorted( set(x for values in feature_names.values() for x in values ))

    return result, feature_names




### TRAINING UTILS ###

def load_and_concat(files_dict):
    frames = []

    for key, files in files_dict.items():
        for f in sorted(files):
            df = pd.read_pickle(f)

            df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) and len(col) == 2 else col for col in df.columns]
            df = df.add_prefix(f"{key}_")
            frames.append(df)

    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def evaluate_model(model, x_test, y_test, params):
    """Evaluate model for both binary and multiclass classification."""
    pred = model.predict(x_test)
    prob = model.predict_proba(x_test)

    unique_classes = np.unique(y_test)

    if len(unique_classes) == 2:
        # --- Binary case ---
        prob_pos = prob[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, prob_pos)

        return {
            'roc_auc': roc_auc_score(y_test, prob_pos),
            'pr_auc': auc(recall, precision),
            'recall': recall_score(y_test, pred, zero_division=0),
            'precision': precision_score(y_test, pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, pred),
            'f1': f1_score(y_test, pred, zero_division=0),
            'params' : params
        }

    else:
        # --- Multiclass case ---
        return {
            'roc_auc': roc_auc_score(y_test, prob, multi_class='ovr'),
            'pr_auc': np.nan,  # not well-defined for multiclass
            'recall': recall_score(y_test, pred, average='macro', zero_division=0),
            'precision': precision_score(y_test, pred, average='macro', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, pred),
            'f1': f1_score(y_test, pred, average='macro', zero_division=0),
            'params' : params
        }

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, balanced_accuracy_score,
    matthews_corrcoef, f1_score, fbeta_score
)

def compute_classification_metrics(true, pred_scores):
    precision, recall, _ = precision_recall_curve(true, pred_scores)
    pr_auc = auc(recall, precision)  # area under PR curve
    minrp = np.minimum(precision, recall).max()
    roc_auc = roc_auc_score(true, pred_scores)
    pred_binary = (pred_scores >= 0.5).astype(int)

    return {
        'auroc': roc_auc,
        'auprc': pr_auc,
        'minrp': minrp,
        'precision': precision_score(true, pred_binary, zero_division=0),
        'recall': recall_score(true, pred_binary, zero_division=0),
        'balanced_acc': balanced_accuracy_score(true, pred_binary),
        'mcc': matthews_corrcoef(true, pred_binary),
        'f1': f1_score(true, pred_binary),
        'f2': fbeta_score(true, pred_binary, beta=2, zero_division=0),
    }


