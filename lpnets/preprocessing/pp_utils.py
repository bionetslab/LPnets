import numpy as np
import pickle
import os
from sklearn.model_selection import StratifiedKFold


def check_no_overlap(train_ids, val_ids, test_ids):
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert train_set.isdisjoint(val_set), "Overlap between Train and Validation sets!"
    assert train_set.isdisjoint(test_set), "Overlap between Train and Test sets!"
    assert val_set.isdisjoint(test_set), "Overlap between Validation and Test sets!"

def print_split_stats(cohort, splits):
    for split_name, ids in splits.items():

        y = cohort.loc[cohort['hadm_id'].astype(str).isin(ids), 'label']

        print(f"{split_name}: n={len(y)}")

        # binary stats
        if y.nunique() <= 2:
            pos_rate = y.mean()
            print(f"  positives: {y.sum()}, rate: {pos_rate:.3f}")

        # multiclass stats
        else:
            counts = y.value_counts()
            for cls, cnt in counts.items():
                print(f"  class {cls}: {cnt} ({cnt/len(y):.3f})")


def compute_splits(cohort, OUT_PATH):

    all_ids = cohort['hadm_id'].astype(str).unique()

    np.random.seed(42)
    np.random.shuffle(all_ids)

    fold_size = len(all_ids) // 3

    # tests sets in CV
    folds = [
        all_ids[0:fold_size],
        all_ids[fold_size:2 * fold_size],
        all_ids[2 * fold_size:]
    ]

    for i in range(5):
        test_ids = folds[i]

        train_valid_ids = np.concatenate([folds[j] for j in range(3) if j != i])
        np.random.shuffle(train_valid_ids)

        bp = int(0.8 * len(train_valid_ids))
        train_ids = train_valid_ids[:bp]
        val_ids = train_valid_ids[bp:]

        print(f"\nFOLD {i}")
        print_split_stats(cohort, {"Train": train_ids, "Val": val_ids, "Test":  test_ids})

        check_no_overlap(train_ids, val_ids, test_ids)

        with open(os.path.join(OUT_PATH, f'fold_{i}.pkl'), 'wb') as f:
            pickle.dump([train_ids, val_ids, test_ids], f)


def compute_splits_stf(cohort, OUT_PATH, n_splits=5):
    X = cohort['hadm_id']
    y = cohort['label']

    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        test_ids = X.iloc[test_idx].astype(str).values

        train_valid_X = X.iloc[train_idx]
        train_valid_y = y.iloc[train_idx]

        # Stratified 80/20 split inside train/val
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        train_inner_idx, val_inner_idx = next(inner_skf.split(train_valid_X, train_valid_y))

        train_ids = train_valid_X.iloc[train_inner_idx].astype(str).values
        val_ids = train_valid_X.iloc[val_inner_idx].astype(str).values

        check_no_overlap(train_ids, val_ids, test_ids)

        print_split_stats(cohort, {"Train": train_ids, "Val": val_ids, "Test": test_ids})

        with open(os.path.join(OUT_PATH, f'fold_{i}.pkl'), 'wb') as f:
            pickle.dump([train_ids, val_ids, test_ids], f)