from lpnets.ml_training.ml_utils import evaluate_model
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score

from lpnets.ml_training.ml_factory import ModelFactory
from lpnets.ml_training.ml_features import FeatureEngineer


SPLITS = {"dev": ["train", "val", "test"], "final": ["train", "test"]}


class MLTrainer:

    def __init__(self, args, stored_outputs):

        self.base_path = args.output_path
        self.og_path = args.original_data_path
        self.args = args
        self.stored_outputs = stored_outputs

        # ---- config ----
        self.model_key = args.model

        self.check_config()
        self.set_output_path()

        # ---- data ----
        self.engineer = FeatureEngineer(args, stored_outputs)
        self.prepare_features()

        # ---- labels ----
        self.y = self.engineer.load_targets()

        # ---- factory ----
        self.factory = ModelFactory(cohort_name=args.cohort)
        self.mlconfig = self.factory.get_item(self.model_key)
        self.param_grid = self.factory.get_grid(self.model_key)


    # ------------------------

    def check_config(self):
        if not self.args.significant:
            self.args.threshold = "NA"
            if not self.args.zscores:
                self.args.zmode = "NA"

    # ------------------------

    def set_output_path(self):
        self.res_path = (
            self.base_path
            / ("sig_edges" if self.args.significant else "all_edges")
            / ("zscores" if self.args.zscores else "edgescores")
            / self.args.zmode
            / self.args.threshold
            / ("tuned" if self.args.hptune else "default")
        )
        self.res_path.mkdir(parents=True, exist_ok=True)


    # ------------------------

    def prepare_features(self):

        self.metrics_final, self.names = self.engineer.compute_features("final")

        if self.args.hptune:
            self.metrics_dev, self.names_dev = self.engineer.compute_features("dev")


    # ------------------------
    
    def save_splits(self, comb_name, X_train, y_train, X_test, y_test):
        out_dir = self.res_path / "splits" / comb_name
        out_dir.mkdir(parents=True, exist_ok=True)

        train_df = X_train.copy()
        train_df["target"] = y_train

        test_df = X_test.copy()
        test_df["target"] = y_test

        train_df.to_parquet(out_dir / "train.parquet")
        test_df.to_parquet(out_dir / "test.parquet")

    # ------------------------

    def save_importance(self, model, feature_names, comb_name):
        importances = None
        if hasattr(model, "get_feature_importance"):
            importances = model.get_feature_importance()
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        fi = None
        if importances is not None:
            fi = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False)

            fi.to_csv(self.res_path / f"{self.model_key}_{comb_name}_feat_imp.csv", index=False)


    def train_comb(self):

        results = {}

        for name_comb in self.names.keys():
            print(f"Training combination: {name_comb}")
            results[name_comb] = self.train_single(name_comb)

        pd.DataFrame(results).T.to_csv(
            self.res_path / f"{self.model_key}_results.csv"
        )

        print(f"Final results saved in {self.res_path}.")


    # ------------------------

    def train_single(self, comb_name):

        def prepare_split(metrics, features, split):
            X = metrics[split][features].fillna(0)
            y = self.y.loc[X.index].values.ravel()
            return X, y

        # ---- hyperparams ----
        if self.args.hptune:
            best_params = self.grid_search(comb_name)
            model, sampler = self.factory.create(self.model_key, mode="custom", params_override=best_params)
        else:
            best_params = {}
            model, sampler = self.factory.create(self.model_key, mode="empty")

        # ---- data ----
        X_train, y_train = prepare_split(self.metrics_final, self.names[comb_name], "train")
        X_test, y_test = prepare_split(self.metrics_final, self.names[comb_name], "test")

        self.save_splits(comb_name, X_train, y_train, X_test, y_test)

        # ---- model via factory ----

        if sampler:
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        model.fit(X_train, y_train)
        results = evaluate_model(model, X_test, y_test, best_params)

        # ---- feature importance ----

        self.save_importance(model, X_train.columns, comb_name)

        print(results)
        return results

    
    # ------------------------

    def grid_search(self, comb_name):

        def prepare_split(metrics, features, split):
            X = metrics[split][features].fillna(0)
            y = self.y.loc[X.index].values.ravel()
            return X, y

        X_train, y_train = prepare_split(self.metrics_dev, self.names_dev[comb_name], "train")
        X_val, y_val = prepare_split(self.metrics_dev, self.names_dev[comb_name], "val")

        best_score = float("-inf")
        best_params = None

        for params in ParameterGrid(self.param_grid):

            print(f"\nConfig : {params}")

            model, sampler = self.factory.create(self.model_key, mode="custom", params_override=params)

            X_tmp, y_tmp = X_train, y_train

            if sampler:
                X_tmp, y_tmp = sampler.fit_resample(X_tmp, y_tmp)

            model.fit(X_tmp, y_tmp)
            preds = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, preds)

            if score > best_score:
                best_score = score
                best_params = params

        print("\nBest params:", best_params)
        return best_params
