import yaml
from pathlib import Path
from lpnets.config.constants import LPNETS
from catboost import CatBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


class ModelFactory:

    def __init__(
        self,
        cohort_name=None,
        best_config_path= LPNETS / "config" / "ml_params_best.yaml",
        grid_config_path= LPNETS / "config" / "ml_params.yaml",
        random_state=42
    ):
        self.random_state = random_state
        self.cohort = cohort_name

        # ---- load YAMLs ----
        self.best_cfg = self._load_yaml(best_config_path)
        self.grid_cfg = self._load_yaml(grid_config_path)

        # ---- registry ----
        self.registry = {
            "catboost_under": {
                "name": "CatBoost",
                "model": CatBoostClassifier,
                "sampler": RandomUnderSampler,
            },
            #"rf_under": {
            #    "name": "Random Forest",
            #    "model": RandomForestClassifier,
            #    "sampler": RandomUnderSampler,
            #},
            "xgb_under": {
                "name": "Xgboost",
                "model": XGBClassifier,
                "sampler": RandomUnderSampler,
            },
            "rf_balanced": {
                "name": "Random Forest",
                "model": BalancedRandomForestClassifier,
                "sampler": None,
            },
        }

        # ---- attach configs to registry ----
        self._attach_configs()

    # ---------- helpers ----------

    def _load_yaml(self, path): # to move to utils
        with open(Path(path)) as f:
            return yaml.safe_load(f)

    def _attach_configs(self):

        for key, cfg in self.registry.items():

            model_name = cfg["name"]

            # ---- best config (dataset-specific) ----
            if self.cohort and model_name in self.best_cfg:
                cfg["best"] = self.best_cfg[model_name].get(self.cohort)
            else:
                cfg["best"] = None

            # ---- default fallback ----
            if cfg["best"] is None:
                cfg["default"] = self.grid_cfg[model_name]["default"]
            else:
                cfg["default"] = cfg["best"]

            # ---- grid ----
            cfg["grid"] = self.grid_cfg[model_name]["grid_search"]

    # ---------- public API ----------

    def create(self, model_key, mode="default", params_override=None):

        cfg = self.registry[model_key]

        # ---- select params ----
        if mode == "empty":
            params = {"n_estimators":500}
            #params = {}
        elif mode == "default":
            params = cfg["default"]
        elif mode == "best":
            params = cfg["best"] 
        elif mode == "custom" and params_override:
            params = {**params_override}
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ---- build model ----
        model = cfg["model"](
            random_state=self.random_state,
            **params,
            verbose=False
        )

        # ---- build sampler ----
        sampler = None
        if cfg["sampler"] is not None:
            sampler = cfg["sampler"](random_state=self.random_state)

        return model, sampler

    def get_grid(self, model_key):
        return self.registry[model_key]["grid"]
    
    def get_item(self, model_key,):
        return self.registry[model_key]