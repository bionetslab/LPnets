import numpy as np
from scipy import stats
from lpnets.features.feature_computer import FeatureComputer
import h5py
from pathlib import Path

class GraphFilter:

    def __init__(self, config, vec_path=None, dict_data=None):
        self.vec_path = vec_path
        self.dict_data = dict_data
        self.config = config

        self.load_data()
        self.final_vec = self.apply_config()
        self.gmc = FeatureComputer(self.all_nodes, self.ids, self.final_vec)
        
    def load_data(self):

        if self.dict_data is not None:
            self.all_nodes = self.dict_data["all_nodes"]
            self.ids = self.dict_data["ids"]
            self.edge_vec = self.dict_data["edges"]

        elif self.vec_path is not None:
            path = Path(self.vec_path)

            if path.suffix == ".h5":
                self.load_vec_h5()
            elif path.suffix == ".npz":
                self.load_vec()
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")

        else:
            raise ValueError("Missing data: provide either vec_path or dict_data")

    def load_vec(self):

        loaded = np.load(self.vec_path, allow_pickle=True)
        data = loaded["data"].item()

        self.all_nodes = data["all_nodes"]
        self.ids = data["ids"]
        self.edge_vec = data["edges"]

    def load_vec_h5(self):

        with h5py.File(self.vec_path, "r") as f:

            # ---- load metadata ----
            self.all_nodes = [
                x.decode("utf-8") for x in f.attrs["all_nodes"]
            ]

            # ---- load ids ----
            self.ids = {
                split: [
                    x.decode("utf-8") for x in f["ids"][split][:]
                ]
                for split in f["ids"]
            }

            # ---- load edge vectors ----
            self.edge_vec = {
                key: f[key][:]
                for key in f.keys()
                if key != "ids"
            }

            print("loaded data H5")

            for split in self.edge_vec:
                print(split, self.edge_vec[split].shape, len(self.ids[split]))

    def apply_config(self):

        if not self.config.significant and not self.config.zscores:
            return self.edge_vec

        self.compute_zscores()

        if not self.config.significant and self.config.zscores:
            return self.zscores
        
        self.apply_threshold()

        if self.config.zscores:
            return self.zscores
        else:
            return self.edge_vec


    def compute_zscores(self):

        train_edges = self.edge_vec["train"]

        if self.config.zmode == "graphwise":
            mean = np.nanmean(train_edges)
            std = np.nanstd(train_edges, ddof=1)

        elif self.config.zmode == "edgewise":
            mean = np.nanmean(train_edges, axis=0)
            std = np.nanstd(train_edges, axis=0, ddof=1)

        else:
            raise ValueError(f"Unknown zmode: {self.config.zmode}")

        std = np.where((std == 0) | np.isnan(std), 1.0, std)

        self.zscores = { split: (edges - mean) / std for split, edges in self.edge_vec.items() }


    def apply_threshold(self):

        if self.config.threshold == "p_val":
            thresh = stats.norm.isf(self.config.p_value / 2)

        elif self.config.threshold == "min":
            # ensure at least one non-masked value per row
            row_max = np.nanmax(np.abs(self.zscores["train"]), axis=1)
            thresh = np.nanmin(row_max)

        else:
            raise ValueError(f"Unknown threshold: {self.config.threshold}")
        
        for split, z in self.zscores.items():
        
            mask = np.abs(z) >= thresh
            self.zscores[split][~mask] = np.nan
            self.edge_vec[split][~mask] = np.nan


    def compute_all_features(self, prefix=None):
        all_metrics_dict = {}
        for feature_type in self.config.features:
            all_metrics_dict[feature_type] = self.gmc.compute_feature_type(feature_type, prefix)
        return all_metrics_dict
    
