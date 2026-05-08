from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy import stats
from lpnets.features.feature_computer import FeatureComputer
from lpnets.features.graph_metric_computer import GraphMetricComputer
from lpnets.edges.edge_utils import *
import pickle
from lpnets.edges.aggregation_utils import (
    aggregate_pcc,
    aggregate_napy_pcc,
    aggregate_pcc_welford,
    aggregate_napy_pcc_welford,
    Aggregator
)


class EdgeMethod(ABC):
    """Base class for single-sample network construction methods."""

    def __init__(self, args):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Rows = admissions or (admission x time) if MultiIndex, columns = features
        ids : dict
            {'train': list_of_indices, 'val': list_of_indices, , 'test': list_of_indices}
        edge_agg_method : str
            Aggregation method, e.g., "PCC", "MI", "LMM"
        mask_df : pd.DataFrame
        """
        self.args = args
        self.is_TS3 = self.args.time_strategy == "TS3"
        self.ids = self.args.ids
        self.significant = self.args.significant
        self.p_value = self.args.p_value
        self.zscore_threshold = stats.norm.isf(self.p_value / 2)
        self.zscores = self.args.zscores
        self.agg_method = args.agg_method
            
    def load_data(self, data, mask_df = None):
        self.data = data
        self.mask_df = mask_df
        self.all_nodes = list(self.data.columns)
        self.n_edges = int(len(self.all_nodes) * (len(self.all_nodes) - 1) / 2)
        self.iu = np.triu_indices(len(self.all_nodes), k=1)
        self.training_data = self.get_subset(self.data, self.ids.get("train", []))
        self.n_train = len(self.training_data)
        dispatch = {
            "PCC":              aggregate_pcc,
            "napyPCC":          aggregate_napy_pcc,
            "PCC_Welford":      aggregate_pcc_welford,
            "napyPCC_Welford":  aggregate_napy_pcc_welford,
        }
        self.aggregator = Aggregator(dispatch[self.agg_method], self.training_data)
        self.train_agg = self.aggregator.corrcoef()
        print("\nData loaded for edge construction.")
  
    
    def apply_mask(self, edge_matrix, sample_mask=None):
        if sample_mask is None:
            return edge_matrix
        
        sample_mask = sample_mask[self.all_nodes].values
        keep_mask = np.outer(~sample_mask, ~sample_mask)
        edge_matrix[~keep_mask] = np.nan

        return edge_matrix


    def compute_aggregate(self, exclude_sample=None, include_sample=None) -> np.ndarray:
        if exclude_sample is not None:
            return self.aggregator.corrcoef_without_sample(exclude_sample)
        elif include_sample is not None:
            return self.aggregator.corrcoef_with_sample(include_sample)
        else:
            return self.aggregator.corrcoef()
        

    def get_subset(self, data, ids):
        # this might change depending on the input, for now
        return data[data.index.get_level_values("hadm_id").isin(ids)]
    

    @abstractmethod
    def compute_edges_adm(self, aggregate_network: pd.DataFrame, perturbed_network: pd.DataFrame, sub_data: pd.DataFrame, idx: int):
        """
        Compute edges for a single admission/sample.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def compute_edges_train(self):
        """Compute edges for training set."""
        train_indices = self.ids.get("train", [])
        if not len(train_indices):
            raise ValueError("No training indices provided for train split.")
        
        agg_all = self.train_agg
        edges_vec = np.zeros((len(train_indices), self.n_edges))

        for id, idx in enumerate(train_indices):
            sample = self.get_subset(self.data, [idx])
            agg_minus_q = self.compute_aggregate(exclude_sample=sample)
            edge_scores = self.compute_edges_adm(agg_all, agg_minus_q, self.n_train, idx, sample_mask=self.mask_df.loc[idx])
            edges_vec[id] = edge_scores

        # compute global statistics only on the training set
        self.global_mean = np.nanmean(edges_vec)
        self.global_std = np.nanstd(edges_vec, ddof=1)
        
        return edges_vec
    
    def compute_edges_eval(self, split_name):
        """Compute edges for evaluation sets (validation or test)."""
        eval_indices = self.ids.get(split_name, [])
        if not len(eval_indices):
            raise ValueError(f"No indices provided for split '{split_name}'.")

        agg_minus_q = self.train_agg
        edges_vec = np.zeros((len(eval_indices), self.n_edges))

        for id, idx in enumerate(eval_indices):
            sample = self.get_subset(self.data, [idx])
            sub_data = pd.concat([self.training_data, sample], axis=0, ignore_index=False)
            agg_all = self.compute_aggregate(include_sample=sample)
            edge_scores = self.compute_edges_adm(agg_all, agg_minus_q, self.n_train + 1, idx, sub_data, self.mask_df.loc[idx])
            edges_vec[id] = edge_scores

        return edges_vec
    

    def compute_edges_all(self):
        """Compute edges for all train, val and test splits."""
        
        #final_edge_dict = {}
        #final_edge_sparse = {}
        
        final_edge_vec = {}
        final_edge_vec["train"] = self.compute_edges_train().astype(np.float32)

        for split_name in self.ids:
            if split_name == "train":
                continue
            final_edge_vec[split_name] = self.compute_edges_eval(split_name).astype(np.float32)

        #for split_name, ids in self.ids.items():
            #final_edge_dict[split_name] = edge_vec_to_dict(final_edge_vec[split_name], ids, self.all_nodes)
            #final_edge_sparse[split_name] = edge_vec_to_sparse(final_edge_vec[split_name], ids, self.all_nodes)
            
        #self.final_edge_dict = final_edge_dict
        #self.final_edge_sparse = final_edge_sparse
        
        self.final_edge_vec = final_edge_vec
        print("\nEdges computed")
        
        for k, v in final_edge_vec.items():
            print(k, v.nbytes / 1e9, "GB")

    def apply_significance_zscore(self, split_vec): # moved to graph filter

        if self.global_std == 0 or np.isnan(self.global_std):
            print("Cannot compute z-score, defaulting to edges without significance.")
            return split_vec.copy()

        zscore_vec = (split_vec - self.global_mean) / self.global_std

        signif_mask = np.abs(zscore_vec) >= self.zscore_threshold

        if self.zscores:
            zscore_vec[~signif_mask] = np.nan
            return zscore_vec
        else:
            split_vec[~signif_mask] = np.nan
            return split_vec             


    def export_flabnet_format(self,feature_type):
        """Format edge outputs for the FlabNet pipeline."""

        gmc = FeatureComputer(self.all_nodes, self.ids, self.final_edge_vec)
        metrics_dict = gmc.compute_feature_type(feature_type)
        return metrics_dict


    def export_gnn_format(self):
        """Format edge outputs for GNN input (PyTorch Geometric Data objects)."""
        data_list_dict = {}

        for split_name, split_data in self.final_edge_dict.items():
            data_list_dict[split_name] = build_graph_list(split_data)

        return data_list_dict
    