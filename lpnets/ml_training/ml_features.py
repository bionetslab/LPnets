import pandas as pd
from lpnets.features.graph_filter import GraphFilter

FILES = {"dev": "edge_vec_dev.h5", "final": "edge_vec.h5"}
SPLITS = {"dev": ["train", "val", "test"], "final": ["train", "test"]}


class FeatureEngineer:

    def __init__(self, args, stored_outputs=None):
        self.args = args
        self.config = args
        self.og_path = args.original_data_path
        self.stored_outputs = stored_outputs


        # optional suffix handling
        self.suffs = {"dev": "", "final": ""}

        # folders (if loading from disk)
        self.fnames = getattr(args, "feature_folders", [])


    def load_targets(self):
        df = pd.read_parquet(self.og_path / "y.parquet")
        return df["label"]
    

    def compute_features(self, stage):

        feature_types = self.config.features

        all_features, feature_names = self.load_original_data(stage)
        feature_names.update({feat: [] for feat in feature_types})

        for prefix, ep in self._iterate_sources(stage):
            self._process_source(ep, prefix, feature_types, all_features, feature_names)

        all_features = {s: pd.concat(dfs, axis=1) for s, dfs in all_features.items()}
        feature_names = {k: sorted(set(v)) for k, v in feature_names.items()}
        feature_names["all"] = sorted({c for v in feature_names.values() for c in v})
    
        return all_features, feature_names


    def load_original_data(self, stage):

        splits = SPLITS[stage]
        features = {split: [] for split in splits}
        names = {}

        for split in splits:
            df = pd.read_parquet(self.og_path / stage / f"{split}.parquet")
            df.columns = [
                f"{c[0]}_{c[1]}" if isinstance(c, tuple) and len(c) == 2 else c
                for c in df.columns
            ]

            features[split].append(df)

        if splits:
            names["original"] = list(df.columns)

        return features, names

    
    def _iterate_sources(self, stage):

        # ---- CASE 1: in-memory ----
        if self.stored_outputs is not None:
            for prefix, dict_data in self.stored_outputs.get(stage, {}).items():
                yield prefix, GraphFilter(config=self.config, dict_data=dict_data)

        # ---- CASE 2: disk ----
        else:
            for folder in self.fnames:
                path = folder / FILES[stage]
                if not path.exists():
                    continue

                yield folder.name, GraphFilter(config=self.config, vec_path=path)


    def _process_source(self, ep, prefix, feature_types, all_features, feature_names):

        for feat in feature_types:

            feat_dict = ep.gmc.compute_feature_type(feat, prefix=prefix)

            for split, df in feat_dict.items():
                all_features[split].append(df)
                feature_names[feat].extend(df.columns)
