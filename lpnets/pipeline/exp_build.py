from types import SimpleNamespace
from lpnets.edges.edge_methods import LIONESS, SSN, SWEET, SWEET_OLD
from lpnets.datasets.pn_dataset import PNDatasetFromSSN
import numpy as np
import h5py
import time
from lpnets.datasets.data_utils import *

METHOD_MAP = {
    "SSN": SSN,
    "LIONESS": LIONESS,
    "SWEET": SWEET,
    "SWEET_OLD": SWEET_OLD,
}

# modelled after Julia's graph constructor

class ExperimentRunner:
    """Base runner. Subclasses implement run(). Comments short."""

    def __init__(self, args: SimpleNamespace, dataset: PNDatasetFromSSN):
        add_output_paths(args)
        self.args = args
        self.output_path = args.output_path
        self.og_path = args.original_data_path
        self.dataset = dataset
        self.builder_cls = METHOD_MAP[args.edge_method]
        self.save_type = args.stage
        self.stored_outputs = {}
        
        self._strategy_map = {
            "TS0": self._run_ts0,
            "TS1": self._run_ts1,
            "TS2": self._run_ts2,
            "TS3": self._run_ts3,
        }

        print(f"\nNumber of datasets {len(self.dataset.prepared_datasets)}")

    def build(self, agg_df, mask_df):
        builder = self.builder_cls(self.args)
        builder.load_data(agg_df, mask_df)
    
        start = time.perf_counter()
        builder.compute_edges_all()
        end = time.perf_counter()
    
        print(f"Edge computation in {(end - start):.2f} seconds")
    
        return builder

    def run(self):
        if self.args.time_strategy not in self._strategy_map:
            raise ValueError(f"Unknown time strategy: {self.args.time_strategy}")
        self._strategy_map[self.args.time_strategy]()
        return self.stored_outputs

    def _run_ts0(self):
        print("\nStart running TS0.")
        agg_df, mask_df = self.dataset.prepared_datasets[0]
        builder = self.build(agg_df, mask_df)
        self.save_artifacts("static" , builder)

    def _run_ts1(self):
        print("\nStart running TS1.")
        for t, (agg_df, mask_df) in enumerate(self.dataset.prepared_datasets):
            builder = self.build(agg_df, mask_df)
            self.save_artifacts(self.args.stats[t], builder)

    def _run_ts2(self):
        print("\nStart running TS2.")
        for t, (agg_df, mask_df) in enumerate(self.dataset.prepared_datasets):
            builder = self.build(agg_df, mask_df)
            self.save_artifacts(f"tw_{t}", builder)

    def _run_ts3(self):
        print("\nStart running TS3.")
        agg_df, mask_df = self.dataset.prepared_datasets[0]
        mask_df = mask_df.groupby(level="hadm_id").first() # TODO even if first is correct in this setting (we impute time series) .any() is more suitable
        builder = self.build(agg_df, mask_df)
        self.save_artifacts("tw_agg", builder)

    def compute_original_data(self):
        if self.args.time_strategy == "TS0":
            self.og_data = self.dataset.prepared_datasets[0][0]
        else:
            self.og_data = self.dataset.get_original_timeseries()


    def check_original_data(self):

        base_path = self.og_path / self.save_type

        for split_name in self.args.ids.keys():
            file_path = base_path / f"{split_name}.parquet"
            if not file_path.exists():
                return False

        y_path = self.og_path / "y.parquet"
        if not y_path.exists():
            return False

        return True


    def save_original_data(self):

        if self.check_original_data():
            print("\nAll original data files already exist. Skipping recomputation.")
            return

        self.compute_original_data()

        out_dir = self.og_path / self.save_type
        out_dir.mkdir(parents=True, exist_ok=True)

        for split_name, ids in self.args.ids.items():
            self.og_data.loc[ids].to_parquet(
                out_dir / f"{split_name}.parquet"
            )

        self.dataset.y.to_frame().to_parquet(self.og_path / "y.parquet")

        print(f"\nOriginal data saved in {out_dir}.")


    def save_artifacts(
        self,
        out_folder,
        builder,
        keep_memory: bool = True,
        save_edge_vec: bool = False,
        save_edge_h5: bool = False,
    ):

        out = self.output_path / self.save_type / out_folder
        out.mkdir(parents=True, exist_ok=True)
        print(f"Output folder created at {out}")

        # -------- memory cache --------
        if keep_memory:
            self._save_in_memory(out_folder, builder)

        # -------- HDF5 --------
        if save_edge_h5:
            self._save_h5(out, builder)

        # -------- NPZ --------
        if save_edge_vec:
            self._save_npz(out, builder)



    def _save_in_memory(self, out_folder, builder):

        self.stored_outputs[out_folder] = {
            "all_nodes": builder.all_nodes,
            "ids": builder.ids,
            "edges": builder.final_edge_vec
        }

        print(f"Outputs in memory ({self.save_type})")


         
    def _save_h5(self, out, builder):

        with h5py.File(out / f"edge_vec.h5", "w") as f:
            
            # ---- store metadata (as strings) ----
            f.attrs["all_nodes"] = np.array(builder.all_nodes, dtype="S")
            # ---- store ids per split ----
            ids_grp = f.create_group("ids")
            for split, id_list in builder.ids.items():
                ids_grp.create_dataset(
                    split,
                    data=np.array(id_list, dtype="S")
                )
            # ---- store each split as dataset ----
            for split, arr in builder.final_edge_vec.items():
                f.create_dataset(
                    split,
                    data=arr,
                    compression="gzip",
                    chunks=True
                )
        print("Edge vector data saved (HDF5)")


    def _save_npz(self, out, builder):

        np.savez_compressed(
            out / "edge_vec.npz",
            all_nodes=builder.all_nodes,
            ids=builder.ids,
            edges=builder.final_edge_vec
        )

        print("Edge vector data saved (NPZ)")





        # # TO CHECK

        # if save_edge_list: #TODO check if it has been computed
            
        #     for split_name, edges in builder.final_edge_dict.items():
        #         with open(out / f"{split_name}_edge_dict.pkl", "wb") as f:
        #             pickle.dump(edges, f)
        #     print("Edge list saved")

        # if save_metrics:
        #     for feature_type in self.args.features:
        #         (out / f"{feature_type}_features").mkdir(parents=True, exist_ok=True)
        #     for feature_type in self.args.features:
        #         metrics_dict = builder.export_flabnet_format(feature_type)
        #         for split_name, metrics in metrics_dict.items():
        #             metrics.to_pickle(out / f"{feature_type}_features"/ f"{split_name}_graph_metrics.pkl")
        #     print("Metrics saved")

        # if save_gnn:
        #     data_list_dict = builder.export_gnn_format()
        #     for split_name, data_list in data_list_dict.items():
        #         data_list.to_pickle(out / f"{split_name}_data_list.pkl")
        #     print("Data list saved")    

