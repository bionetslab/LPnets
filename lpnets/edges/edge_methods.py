from lpnets.edges.edge_builder import EdgeMethod
import numpy as np
from lpnets.edges.aggregation_utils import (
    aggregate_pcc,
    aggregate_napy_pcc)


# Example subclasses
class SSN(EdgeMethod):
    def compute_edges_adm(self, agg_all, agg_minus_q, N, idx, sub_data=None, sample_mask=None):

        # --- SSN formula --- 
        edge_matrix = agg_all - agg_minus_q  #TODO check for nan here (you can still have nan in napy if one column is all nan)
        edge_matrix = self.apply_mask(edge_matrix, sample_mask)
        return edge_matrix[self.iu]


class LIONESS(EdgeMethod):
    def compute_edges_adm(self, agg_all, agg_minus_q, N, idx, sub_data=None, sample_mask=None):

        # --- LIONESS formula --- 
        edge_matrix = N * (agg_all - agg_minus_q) + agg_minus_q
        edge_matrix = self.apply_mask(edge_matrix, sample_mask)
        return edge_matrix[self.iu]


class SWEET_OLD(EdgeMethod):
    def __init__(self, args):
        super().__init__(args)
        self.weights_train = None


    def load_data(self, data, mask_df=None):
        super().load_data(data, mask_df)
        self.weights_train = self.compute_weights(self.training_data)


    def compute_weights(self, data):
        """Compute sample weights for all admissions."""
        k = 0.1

        if self.is_TS3:
            wide = data.unstack(level='bin')   # hadm × (feature × day))
            values = wide.to_numpy()
            sample_ids = wide.index.tolist()
        else:
            # Standard mode: data has single index (admissions)
            values = data.to_numpy()  # admissions x features                  
            sample_ids = data.index.tolist()
        # Compute correlation-based (samples x samples correlation) sample weights (original SWEET method)
        #TODO: add and use self.imputed?
        if self.agg_method in ("napyPCC", "napyPCC_Welford"):
            corr_matrix, _ = aggregate_napy_pcc(values.T)
        else:
            corr_matrix, _ = aggregate_pcc(values.T)
        # Handle single‑sample case
        if values.shape[0] == 1:
            weights = np.array([1.0])
        else:
            N = corr_matrix.shape[0]  # number of samples
            weights = (np.sum(corr_matrix, axis=1) - 1) / (N - 1)
            weights_max, weights_min = np.max(weights), np.min(weights)
            dif = weights_max - weights_min + 0.01
            weights = (weights - weights_min + 0.01) / dif
            weights = weights * k * N

        # Create dictionary mapping sample (admission) IDs to weights
        weights_dict = dict(zip(sample_ids, weights))
        return weights_dict
        

    def compute_edges_adm(self, agg_all, agg_minus_q, N, idx, sub_data=None, sample_mask=None):
        # Idx is from training data; get the weight for this sample from the precomputed training weights
        if sub_data is None:
            w_q = float(self.weights_train[idx])
        # Idx is from evaluation data
        else:
            weights = self.compute_weights(sub_data)
            w_q = float(weights[idx])

        # --- SWEET formula --- 
        edge_matrix = w_q * (agg_all - agg_minus_q) + agg_minus_q
        edge_matrix = self.apply_mask(edge_matrix, sample_mask)
        return edge_matrix[self.iu]

class SWEET(EdgeMethod):
    def __init__(self, args):
        super().__init__(args)
        self.weights_train = None

    def load_data(self, data, mask_df=None):
        super().load_data(data, mask_df)
        self.weights_train = self.compute_weights(self.training_data)

    def compute_weights(self, data):
        """Compute sample weights for all admissions."""
        k = 0.1
        if self.is_TS3:
            wide = data.unstack(level='bin')
            values = wide.to_numpy()
            sample_ids = wide.index.tolist()
        else:
            values = data.to_numpy()
            sample_ids = data.index.tolist()
        #TODO: add and use self.imputed?
        if self.agg_method in ("napyPCC", "napyPCC_Welford"):
            corr_matrix, _ = aggregate_napy_pcc(values.T)
        else:
            corr_matrix, _ = aggregate_pcc(values.T)

        self._weight_values = values
        self._weight_corr_row_sums = np.nansum(corr_matrix, axis=1)   

        if values.shape[0] == 1:
            weights = np.array([1.0])
        else:
            N = corr_matrix.shape[0]
            weights = (self._weight_corr_row_sums - 1) / (N - 1)
            weights_max, weights_min = np.max(weights), np.min(weights)
            dif = weights_max - weights_min + 0.01
            weights = (weights - weights_min + 0.01) / dif
            weights = weights * k * N

        return dict(zip(sample_ids, weights))

    def _pairwise_corr_with_new_sample(self, new_values):
        """Correlate a single new sample against all training samples."""
        train = self._weight_values
        if self.agg_method in ("napyPCC", "napyPCC_Welford"):
            return np.array([
                self._pairwise_corr_nan(new_values, row) for row in train
            ])
        else:
            new_c = new_values - new_values.mean()
            train_c = train - train.mean(axis=1, keepdims=True)
            numer = train_c @ new_c
            denom = np.sqrt(np.sum(train_c**2, axis=1)) * np.sqrt(np.sum(new_c**2))
            return np.where(denom > 0, numer / denom, 0.0)

    @staticmethod
    def _pairwise_corr_nan(a, b):
        """Pearson correlation between two vectors, ignoring pairwise NaNs."""
        valid = ~np.isnan(a) & ~np.isnan(b)
        if valid.sum() < 2:
            return 0.0
        a_v, b_v = a[valid], b[valid]
        a_c, b_c = a_v - a_v.mean(), b_v - b_v.mean()
        denom = np.sqrt(np.sum(a_c**2) * np.sum(b_c**2))
        return (a_c @ b_c) / denom if denom > 0 else 0.0

    def _compute_eval_weight(self, new_values):
        """Get the weight for one eval sample by extending the cached training
        correlation matrix with just N new entries instead of rebuilding it."""
        k = 0.1
        N = self._weight_values.shape[0]

        # Only N new correlations needed, not the full (N+1)×(N+1) matrix
        new_corrs = self._pairwise_corr_with_new_sample(new_values)

        # New sample's raw weight: its row sum (excluding self-corr) / N
        raw_new = np.sum(new_corrs) / N

        # Each training sample's row sum just grows by its corr with the new sample
        raw_train = (self._weight_corr_row_sums + new_corrs - 1) / N

        # Normalize together so min/max are consistent across all N+1 samples
        all_raw = np.append(raw_train, raw_new)
        w_max, w_min = all_raw.max(), all_raw.min()
        dif = w_max - w_min + 0.01
        all_norm = (all_raw - w_min + 0.01) / dif * k * (N + 1)

        # Last entry is the new sample's weight
        return float(all_norm[-1])

    def compute_edges_adm(self, agg_all, agg_minus_q, N, idx,
                          sub_data=None, sample_mask=None):
        if sub_data is None:
            # Training: use precomputed weights
            w_q = float(self.weights_train[idx])
        else:
            # Evaluation: compute weight incrementally
            sample = self.get_subset(self.data, [idx])
            if self.is_TS3:
                vals = sample.unstack(level='bin').to_numpy().ravel()
            else:
                vals = sample.to_numpy().ravel()
            w_q = self._compute_eval_weight(vals)

        # SWEET formula
        edge_matrix = w_q * (agg_all - agg_minus_q) + agg_minus_q
        edge_matrix = self.apply_mask(edge_matrix, sample_mask)
        return edge_matrix[self.iu]