import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from lpnets.edges.edge_utils import *
from lpnets.features.graph_metric_computer import GraphMetricComputer

#TODO understand the computational intensive steps (graph computation, metrics computation?)
class FeatureComputer:

    def __init__(self, all_nodes, ids, final_edge_vec):
        self.all_nodes = all_nodes
        self.top_n_edges = 1500 # updated to 2000 to match omic dimensionality
        self.final_edge_vec = final_edge_vec
        self.top_train_idx = None
        self.edge_col_names = None
        self.ids = ids
        self.compute_edge_sparse()

    def compute_edge_sparse(self):
        self.final_edge_sparse = {}
        for split_name, ids in self.ids.items():
            self.final_edge_sparse[split_name] = edge_vec_to_sparse(self.final_edge_vec[split_name], ids, self.all_nodes)

    def get_edge_names(self):
        n_nodes = len(self.all_nodes)
        iu = np.triu_indices(n_nodes, k=1)
        self.edge_col_names = np.array([f"{self.str_prefix}edge_{self.all_nodes[i]}_{self.all_nodes[j]}" for i, j in zip(iu[0], iu[1])], dtype=object)

    def compute_feature_type(self, feature_type, prefix=None):

        self.str_prefix = "" if prefix is None else f"{prefix}_"

        if feature_type == "node":
            return self.compute_node_all()
        elif feature_type == "edge":
            return self.compute_edge_all()
        elif feature_type == "graph":
            return self.compute_graph_all()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
    # COMPUTE NODE FEATURES

    def compute_node_all(self):
        metrics = {}
        for split_name, split_dict in self.final_edge_sparse.items():
            metrics[split_name] = self._node_split(split_dict)
        return metrics
    
    def _node_split(self, split_dict):
        n = len(self.all_nodes)
        rows = []

        for _, sparse_adj in split_dict.items():
            strength = np.abs(sparse_adj).sum(axis=1).A1 / (n - 1)
            rows.append(strength)

        return pd.DataFrame(
            np.vstack(rows),
            index=split_dict.keys(),
            columns=[f"{self.str_prefix}node_{node}_degree_centrality" for node in self.all_nodes]
        )
    
    # COMPUTE EDGE FEATURES
    
    def compute_edge_all(self):

        self.get_edge_names()
        metrics = {}
        # compute on train first (variance-based selection)
        metrics["train"] = self._edge_train(self.final_edge_vec["train"])
        # compute on other splits using same top indices
        for split_name, split_vec in self.final_edge_vec.items():
            if split_name == "train":
                continue
            metrics[split_name] = self._edge_eval(split_vec, split_name)
        return metrics
    
    def _edge_train(self, split_vec):
        var = np.nanvar(split_vec, axis=0)
        top_idx = np.argsort(var)[::-1][:self.top_n_edges]
        self.top_train_idx = top_idx
        return self._top_edge_df(split_vec, self.ids["train"])

    def _edge_eval(self, split_vec, split_name):
        if self.top_train_idx is None:
            raise RuntimeError("top_train_idx not set. Run training split first.")
        return self._top_edge_df(split_vec, self.ids[split_name])

    def _top_edge_df(self, split_vec, ids=None):

        if self.edge_col_names is None:
            self.get_edge_names()

        top_vec = split_vec[:, self.top_train_idx]
        top_vec[np.isnan(top_vec)] = 0.0 # impute nan here #TODO: consider a nan mask to differentiate zero edge and unobserved
        top_cols = self.edge_col_names[self.top_train_idx]
        return pd.DataFrame(top_vec, index=ids, columns=top_cols)   
    
    # COMPUTE GRAPH FEATURES

    def compute_graph_all(self):
        self.graph_computer = GraphMetricComputer(self.all_nodes)
        
        metrics = {}
        for split_name, split_dict in self.final_edge_sparse.items():
            metrics[split_name] = self._graph_split(split_dict)
        return metrics

    def _graph_split(self, split_dict):
        rows = []
        for sample_id, sparse_adj in split_dict.items():
            G = nx.from_scipy_sparse_array(sparse_adj)
            mapping = {i: node_id for i, node_id in enumerate(self.all_nodes)}
            G = nx.relabel_nodes(G, mapping)
            graph_metrics = self.graph_computer.calculate_graph_metrics(G)
            graph_metrics = {f"{self.str_prefix}{k}": v for k, v in graph_metrics.items()}
            graph_metrics["graph_id"] = sample_id
            rows.append(graph_metrics)
        return pd.DataFrame(rows).set_index("graph_id")