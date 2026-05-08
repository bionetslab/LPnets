import pandas as pd
from sklearn.impute import KNNImputer

from lpnets.datasets.ts_dataset import PNDataset
#from ts_model_training.dataset import TimeSeriesDataset
#from ts_model_training.utils import ids_in_data, set_splits

class PNDatasetFromSSN(PNDataset):
#class PNDatasetFromSSN(TimeSeriesDataset):
    def __init__(self, args):
        super().__init__(args)

        self._strategy_map = {
            "TS0": self._prepare_static,
            "TS1": self._prepare_summary,
            "TS2": self._prepare_per_bin,
            "TS3": self._prepare_stacked,
        }
        self.args.imputed = False
        self.args.has_na = False
        self.preprocess_data()

    def preprocess_data(self):
        # Prepare datasets for edge builders
        self.adjust_time_strategy()
        self.update_impute()
        self.prepare_data()
        self.update_method()
        # Free memory
        #self.cleanup_raw_data()

    def cleanup_raw_data(self):
        if hasattr(self, "data"):
            del self.data
        if hasattr(self, "data_imp"):
            del self.data_imp
        print("\nRaw data cleared")

    # ADDED to generate original data
    def get_original_timeseries(self):
        if not hasattr(self, "data_imp"):
            self.data_imp = self.impute_ts()
        return self.data_imp.reorder_levels(["itemid", "bin"], axis=1).sort_index(axis=1) 

    # Strategy handling
    def adjust_time_strategy(self):

        if self.args.dataset_type == "static" and self.args.time_strategy != "TS0":
            print("\nStatic dataset but temporal strategy → forcing TS0")
            self.args.time_strategy = "TS0"

        elif self.args.dataset_type == "temporal" and self.args.time_strategy == "TS0":
            print("\nTemporal dataset but static strategy → switching to TS1")
            self.args.time_strategy = "TS1"
            
    def update_impute(self):
        if "napy" not in self.args.agg_method:
            self.args.to_impute = True
            print("\nIf missing values, data will be imputed")
        else:
            self.args.to_impute = False


    def update_method(self):
        if "napy" in self.args.agg_method and not self.args.has_na:
            self.args.logger.write(
                "\nNo missing values detected. No need for na-aware test."
            )
            # remove "napy" from method name
            self.args.agg_method = self.args.agg_method.replace("napy", "")
        print(f"\nApplying aggregation method {self.args.agg_method}")

    # TIME STRATEGY IMPLEMENTATION

    def prepare_data(self):
        if self.args.time_strategy not in self._strategy_map:
            raise ValueError(f"Unknown time strategy: {self.args.time_strategy}")
        self.prepared_datasets = self._strategy_map[self.args.time_strategy]()


    def _prepare_static(self):
        df = self.data.pivot(index="hadm_id", columns="itemid", values="value")
        df, mask = self.impute_df(df)
        return [(df, mask)]
    
    def _prepare_summary(self):
        datasets = []
        for method in self.args.stats:
            self.args.logger.write(f"\nPrepare summary dataset with stat {method}")
            df = self.data.groupby(["hadm_id", "itemid"])["value"].agg(method).unstack()
            df, mask = self.impute_df(df)
            datasets.append((df, mask))
        return datasets
    
    def _prepare_per_bin(self):
        self.data_imp = self.impute_ts()
        self.data_imp, self.mask = self.impute_df(self.data_imp)
        datasets = []
        for d in sorted(self.data["bin"].unique()):
            self.args.logger.write(f"\nPrepare dataset for bin {d}")
            df = self.data_imp.xs(d, axis=1, level=0)
            mask = self.mask.xs(d, axis=1, level=0)
            datasets.append((df, mask))
        return datasets
    
    def _prepare_stacked(self):
        self.data_imp = self.impute_ts()
        self.data_imp, self.mask = self.impute_df(self.data_imp)
        return [(self.data_imp.stack("bin"), self.mask.stack("bin"))]

    # TIME SERIES IMPUTATION

    def impute_ts(self):
        
        self.args.logger.write("\nImpute time series with forward and backward fill")
        adms = self.data["hadm_id"].unique()
        items = self.data["itemid"].unique()
        data_imp = (
            self.data.groupby(["hadm_id", "itemid", "bin"])["value"]
            .mean()
            .unstack(level="bin")
            .ffill(axis=1)
            .bfill(axis=1)
            .reindex(pd.MultiIndex.from_product([adms, items], names=["hadm_id", "itemid"]))
            .unstack("itemid")
        )
        return data_imp

    # SAMPLE IMPUTATION

    def impute_df(self, data):

        mask_df = data.isna()

        #if not self.args.impute and mask_df.any().any() and self.args.agg_method == "PCC":
        #    self.args.logger.write("\nCannot use PCC with NaN values. Enabling imputation.")
        #    self.args.impute = True
        if mask_df.any().any():

            self.args.has_na = True

            if self.args.to_impute:# and mask_df.any().any():

                self.args.imputed = True

                self.args.logger.write("\nImpute remaining missing values using KNN imputer fitted on train split.")
                imputer = KNNImputer(n_neighbors=5)
                train_df = data.loc[self.args.ids["train"]]
                imputer.fit(train_df)

                dfs = []
                for ids in self.args.ids.values():
                    sub_df = data.loc[ids]
                    df_imp = pd.DataFrame(imputer.transform(sub_df), index=sub_df.index, columns=sub_df.columns)
                    dfs.append(df_imp)

                full_imp = pd.concat(dfs).sort_index()

                if full_imp.isna().any().any():
                    raise ValueError("agg_df contains NaN values after imputation.")

                return full_imp, mask_df

        return data, mask_df
        