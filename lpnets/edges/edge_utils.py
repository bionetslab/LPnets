import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
import napypi as napy

## FORMAT CONVERSION UTILS ##

def edge_vec_to_dict(edge_vec, ids, nodes):
    iu = np.triu_indices(len(nodes), k=1)
    final_adj_dict = {}
    for id, hadm in enumerate(ids):
        edge_scores = edge_vec[id]
    
        adj_dict = {node: {} for node in nodes}
        for i, j, weight in zip(iu[0], iu[1], edge_scores):
            if np.isnan(weight): #weight == 0 or 
                continue
            node_i = nodes[i]
            node_j = nodes[j]
            adj_dict[node_i][node_j] = weight
            adj_dict[node_j][node_i] = weight
        final_adj_dict[hadm] = adj_dict

    return final_adj_dict

def edge_vec_to_sparse(edge_vec, ids, nodes):
    n_nodes = len(nodes)
    iu = np.triu_indices(n_nodes, k=1)
    sparse_graphs = {}
    for idx, graph_id in enumerate(ids):

        weights = edge_vec[idx]

        mask = ~np.isnan(weights)

        rows = iu[0][mask]
        cols = iu[1][mask]
        data = weights[mask].astype(float)

        # make symmetric
        rows_sym = np.concatenate([rows, cols])
        cols_sym = np.concatenate([cols, rows])
        data_sym = np.concatenate([data, data])

        sparse_adj = coo_matrix((data_sym, (rows_sym, cols_sym)), shape=(n_nodes, n_nodes))

        sparse_graphs[graph_id] = sparse_adj

    return sparse_graphs

def nested_dict_to_sparse(nested_dict, all_nodes):

    """Convert nested dictionary {graph_id: {node1: {node2: weight}}} 
    to a dictionary of sparse adjacency matrices {graph_id: coo_matrix}."""
    # Node-ID to index mapping
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    n_nodes = len(all_nodes)
    adj_matrices = {}

    for graph_id, graph_data in nested_dict.items():
        rows, cols, data = [], [], []
        for node_i, neighbors in graph_data.items():
            i = node_to_idx[node_i]
            for node_j, weight in neighbors.items():
                j = node_to_idx[node_j]
                rows.append(i)
                cols.append(j)
                data.append(float(weight))

        # Make symmetric
        rows_sym = rows + cols
        cols_sym = cols + rows
        data_sym = data + data

        sparse_adj = coo_matrix((data_sym, (rows_sym, cols_sym)), shape=(n_nodes, n_nodes))
        adj_matrices[graph_id] = sparse_adj

    return adj_matrices


def build_graph_list(edge_dicts: dict): #TODO:check if this is the right expo
    """Convert nested adjacency dictionaries into PyTorch Geometric Data objects."""
    data_list = []
    for graph_id, nodes in edge_dicts.items():
        edge_list, edge_weights = [], []

        # Collect all edges
        for src, neighbors in nodes.items():
            for dst, weight in neighbors.items():
                edge_list.append([src, dst])
                edge_weights.append(float(weight))

        # Map node IDs to 0..N-1 indices
        unique_nodes = sorted({n for e in edge_list for n in e})
        node2idx = {node: i for i, node in enumerate(unique_nodes)}

        edge_index = torch.tensor([[node2idx[src], node2idx[dst]] for src, dst in edge_list],dtype=torch.long).t()

        edge_attr = torch.tensor(edge_weights, dtype=torch.float)

        # Node features (identity matrix as default)
        node_feat = torch.eye(len(unique_nodes))

        data = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_attr)
        data.graph_id = graph_id
        data_list.append(data)

    return data_list