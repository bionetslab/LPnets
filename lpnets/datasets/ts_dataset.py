import pandas as pd
import numpy as np
import pickle

from lpnets.datasets.data_utils import *

class PNDataset:
    def __init__(self, args):
        self.args = args
        load_folds(args)
        self.load_data()
        self.get_temporal_data()
        self.get_static_data()
        self.get_target()    
        self.check_temporal()
        #self.preprocess_data() # all preprocessing moved to the inherited class

    def load_data(self):
        # load time series data with dtype specifications
        dtype_spec = { 'subject_id': 'string', 'hadm_id': 'string', 'bin': 'int64', 'day': 'int64', 'itemid': 'string', 'value': 'float64'}
        # load target cohort
        self.cohort = pd.read_csv(self.args.paths["cohort_path"], compression='gzip', dtype=dtype_spec)

        # load data
        self.data = pd.read_csv(self.args.paths["data_path"], dtype=dtype_spec)

        if self.args.subsample is not None and isinstance(self.args.subsample, int):
            self.data = self.data.sample(frac=float(self.args.subsample/100), random_state=42)

        # keep ids found in data
        self.data, self.args.ids, self.args.sup_ids = ids_in_data(self.data, self.args.ids, self.args.logger)

    def get_temporal_data(self):
        
        # select features specified by file
        if "top_features" in self.args.paths:
            feature_file = self.args.paths["top_features"]
            with open(feature_file, 'rb') as f:
                self.selected_features = list(map(str, pickle.load(f)))
            self.data = self.data[self.data["itemid"].isin(self.selected_features)]
            self.args.logger.write('\nFeature selected from file: ' + str(len(self.selected_features)))
            
        # remove features not present in training data
        self.data, del_var = remove_features_not_in_train(self.data, self.args.ids["train"])

        # remove constant features
        self.data, const_var = remove_constant_in_train(self.data, self.args.ids["train"])

        # select features specified by parameter
        self.args.logger.write('\nTemporal features: '+str(self.data.itemid.nunique()))
        self.args.logger.write('\nRemoving variables not in training set: ' + str(del_var))
        self.args.logger.write('\nRemoving constant variables: ' + str(const_var))

        # check again after filtering features
        self.data, self.args.ids, self.args.sup_ids = ids_in_data(self.data, self.args.ids, self.args.logger)


    # TODO: demographic features not used atm, concat to graph features in model training
    def get_static_data(self):

        demo_varis = ["age", "gender"]
        static_data = self.cohort[["hadm_id"] + demo_varis].copy()
        static_data["gender"] = static_data["gender"].astype(str).str.upper().map({"M": 0, "F": 1})
        static_data = static_data.loc[static_data.hadm_id.isin(self.args.sup_ids)]
        self.static_data = static_data.set_index("hadm_id")
        demo_raw = self.static_data.values
        train_mask = self.static_data.index.isin(self.args.ids["train"])

        # normalisation not needed for Catboost but needed for GNN
        demo_means = np.nanmean(demo_raw[train_mask], axis=0, keepdims=True)
        demo_stds  = np.nanstd(demo_raw[train_mask], axis=0, keepdims=True)
        demo_stds = np.where(demo_stds == 0, 1, demo_stds)
        demo_norm = (demo_raw - demo_means) / demo_stds

        # missing indicators (correct direction)
        cols_with_na = self.static_data.columns[self.static_data.isna().any()]
        if len(cols_with_na) > 0:
            missing_indicators = (self.static_data[cols_with_na].isna().astype(int).values)
            demo_norm = np.concatenate([demo_norm, missing_indicators], axis=1)
            demo_raw = np.concatenate([demo_raw, missing_indicators], axis=1)
            demo_raw = np.nan_to_num(demo_raw, nan=0.0)
            demo_norm = np.nan_to_num(demo_norm, nan=0.0)

        #self.demo_means = demo_means
        #self.demo_stds = demo_stds
        self.demo_raw = demo_raw
        self.demo = demo_norm

        self.args.D = self.demo.shape[1]

        self.args.logger.write("\nStatic variables: " + ", ".join(self.static_data.columns))
        self.args.logger.write("\nTotal static features: " + str(self.args.D))


    def get_target(self):

        # only compute target in supervised settings
        self.y = self.cohort[['hadm_id', 'label']].set_index('hadm_id').loc[self.args.sup_ids,'label']

        # compute class weight
        pos_freq = {}
        for split_name, ids in self.args.ids.items():
            y_split = self.y.loc[ids]
            pos_freq[split_name] = safe_pos_freq(y_split)
        
        self.args.logger.write('\n% pos class per split: ' + str({k: round(v, 3) for k, v in pos_freq.items()}))


    def check_temporal(self):
        # compute max number of observations per (hadm_id,itemid)
        maxobs = self.data.groupby(['hadm_id', 'itemid']).size().max()

        self.args.dataset_type = "static" if maxobs == 1 else "temporal"
        self.args.logger.write(f"\nDataset detected as: {self.args.dataset_type}")

        # if temporal set bin column
        if self.args.dataset_type == "temporal":
            if self.args.bin not in self.data.columns:
                raise ValueError("Temporal data detected but bin column not provided.")    
            self.data = self.data.rename(columns={self.args.bin: "bin"})   
   