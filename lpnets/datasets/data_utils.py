import pickle
import numpy as np
from datetime import datetime
import random
from lpnets.config.constants import GRAPH_RESULTS, DATASET_CONFIG

### LOADING UTILS ###

def add_input_paths(args):

    cfg = DATASET_CONFIG[args.cohort]
    base = cfg["base"]

    paths = {
        "data_path": base / cfg["data_file"],
        "cohort_path": base / cfg["cohort_file"],
        "CV_folds_path": base / cfg["CV_folds_path"],
    }

    if "top_features" in cfg:
        paths["top_features"] = base / cfg["top_features"]

    args.paths = paths


def add_output_paths(args):

    sub_dir = "full"

    if args.subset is not None and args.subsample is not None:
        sub_dir = f"sub_{args.subset}_frac_{args.subsample}"
    elif args.subset is not None:
        sub_dir = f"sub_{args.subset}"
    elif args.subsample is not None:
        sub_dir = f"frac_{args.subsample}"

    base_path = (
        GRAPH_RESULTS /
        f'{args.cohort}' /
        f'fold_{args.fold}' /
        sub_dir /
        ("static" if args.bin is None else args.bin)
    )

    # set experiment name to current date if not provided
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime("%d%m%y")

    # output paths
    path = (
        base_path /
        args.pipeline /
        args.exp_name /
        ("imp" if args.imputed else "noimp") /
        args.time_strategy /
        args.edge_method /
        args.agg_method
        #("sig_edges" if args.significant else "all_edges") / 
        #("zscores" if args.zscores else "edgescores")
    )

    #if getattr(args, "significant", False):
    #    path = path / "sig_edges"

    #if getattr(args, "zscores", False):
    #    path = path / "zscores"

    args.output_path = path
    args.original_data_path = base_path / "original_data"


def process_ids(ids):
    ids = np.array(ids)
    # if 1D -> use whole array # this is for icu and omic
    if ids.ndim == 1:
        return ids.astype(str)
    # if 2D -> use second column # this is for mimic and uker cohorts
    elif ids.ndim == 2:
        return ids[:, 1].astype(str)
    else:
        raise ValueError("Unexpected ID array shape")
    

def load_folds(args):

    # load splits
    fold_file = args.paths["CV_folds_path"] / f"fold_{args.fold}.pkl"

    with open(fold_file, "rb") as f:
        train_ids, val_ids, test_ids = pickle.load(f)

    train_ids = process_ids(train_ids)
    val_ids   = process_ids(val_ids)
    test_ids   = process_ids(test_ids)

    # subset (optional)

    if args.subset is not None and isinstance(args.subset, int):
        random.seed(42)
        random.shuffle(train_ids)
        random.shuffle(val_ids)
        random.shuffle(test_ids)
        train_ids = train_ids[:args.subset]
        val_ids   = val_ids[:args.subset // 5]
        test_ids  = test_ids[:args.subset // 2]

    # check overlap (must not happen)

    if np.intersect1d(train_ids, val_ids).size > 0:
        raise ValueError("Overlap between train and val splits.")
    if np.intersect1d(train_ids, test_ids).size > 0:
        raise ValueError("Overlap between train and test splits.")
    if np.intersect1d(val_ids, test_ids).size > 0:
        raise ValueError("Overlap between val and test splits.")

    # no-val mode → merge val into train

    if args.stage == "final":
        train_ids = np.concatenate([train_ids, val_ids])
        val_ids = np.array([], dtype=str)

    # build dictionary and drop empty splits

    ids = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    # remove empty splits
    ids = {k: v for k, v in ids.items() if len(v) > 0}

    # training must exist
    if "train" not in ids:
        raise ValueError("Training split is empty.")

    args.ids = ids



### PREPROCESSING UTILS ###

def ids_in_data(data, ids, logger=None):

    all_ids = []
    for key, values in ids.items():
        all_ids.append(np.array(values, dtype=str))

    # flatten all ids (safe if empty)
    all_ids = np.unique(np.concatenate(all_ids)) if all_ids else np.array([], dtype=str)

    curr_ids = data.hadm_id.unique()
    missing_ids = np.setdiff1d(all_ids, curr_ids)

    # for each element that is not na, get intersections 
    valid_ids = {}
    for key in ids:
        intersected = np.intersect1d(ids[key], curr_ids)
        if len(intersected) > 0:
            valid_ids[key] = intersected

    if not valid_ids:
        raise ValueError("No valid IDs remain after filtering. ")
    if "train" not in valid_ids:
        raise ValueError("Training split is empty after filtering.")

    # concatenate not na
    sup_ts_ids = np.concatenate([v for v in valid_ids.values()])

    if logger is not None:
        logger.write(f"\nTotal ids removed (not in data): {len(missing_ids)}")
        logger.write(f"\n# splits filtered: " + ", ".join(f"{k} : {len(v)}" for k, v in valid_ids.items()))

    # keep only relevant ids in data
    data = data.loc[data.hadm_id.isin(sup_ts_ids)]

    return data, valid_ids, sup_ts_ids


def remove_features_not_in_train(data, train_ids):
    
    train_variables = np.array(data.loc[data.hadm_id.isin(train_ids), 'itemid'].unique(), dtype=str)
    all_variables = np.array(data['itemid'].unique(), dtype=str)
    delete_variables = np.setdiff1d(all_variables, train_variables)

    return data.loc[data['itemid'].isin(train_variables)], delete_variables

def  remove_constant_in_train(data, train_ids):

    variable_variability = data.loc[data.hadm_id.isin(train_ids)].groupby("itemid")["value"].nunique()
    non_constant_variables = variable_variability[variable_variability > 1].index
    constant_variables = variable_variability[variable_variability == 1].index

    return data.loc[data["itemid"].isin(non_constant_variables)], constant_variables.tolist()


def safe_pos_freq(splits):
    if len(splits) == 0:
        return np.nan 
    return splits.sum() / len(splits)


def print_args(args):
    print("Parsed arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")