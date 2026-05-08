import numpy as np
import pandas as pd
from dataclasses import dataclass
import napypi as napy


# State dataclasses

@dataclass
class WelfordState:
    n: int
    mean: np.ndarray
    M2: np.ndarray


@dataclass
class WelfordNaNState:
    n_total: int
    non_nan: np.ndarray
    mean: np.ndarray
    S: np.ndarray
    C: np.ndarray


# Welford helpers (no NaN)

def init_welford(data: np.ndarray):
    data = np.asarray(data, dtype=np.float64)
    n, p = data.shape
    mean = data.mean(axis=0)
    centered = data - mean
    M2 = centered.T @ centered
    return n, mean, M2


def update(n: int, mean: np.ndarray, M2: np.ndarray, sample: np.ndarray):
    arr = np.atleast_2d(np.asarray(sample, dtype=np.float64))
    cur_n, cur_mean, cur_M2 = n, mean, M2
    for x in arr:
        new_n = cur_n + 1
        delta1 = x - cur_mean
        new_mean = cur_mean + delta1 / new_n
        delta2 = x - new_mean
        cur_M2 = cur_M2 + np.outer(delta1, delta2)
        cur_n, cur_mean = new_n, new_mean
    return cur_n, cur_mean, cur_M2


def downdate(n: int, mean: np.ndarray, M2: np.ndarray, sample: np.ndarray):
    arr = np.atleast_2d(np.asarray(sample, dtype=np.float64))
    cur_n, cur_mean, cur_M2 = n, mean, M2
    for x in arr:
        new_n = cur_n - 1
        delta1 = x - cur_mean
        new_mean = (cur_n * cur_mean - x) / new_n
        delta2 = x - new_mean
        cur_M2 = cur_M2 - np.outer(delta1, delta2)
        cur_n, cur_mean = new_n, new_mean
    if np.any(np.diag(cur_M2) < -1e-10):
        print("Warning: downdate produced negative variance.")
    return cur_n, cur_mean, cur_M2


def corrcoef_from_state(n: int, mean: np.ndarray, M2: np.ndarray) -> np.ndarray:
    cov = M2 / (n - 1)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr)
    return np.clip(corr, -1.0, 1.0)


# Welford helpers (NaN-aware, pairwise)

def init_welford_nan(data: np.ndarray):
    data = np.asarray(data, dtype=np.float64)
    n_samples, p = data.shape
    non_nan = np.zeros((p, p))
    mean = np.zeros((p, p))
    S = np.zeros((p, p))
    C = np.zeros((p, p))

    for row in data:
        non_nan, mean, S, C = _update_row_nan(row, non_nan, mean, S, C)

    return n_samples, non_nan, mean, S, C


def _update_row_nan(x: np.ndarray, non_nan: np.ndarray, mean: np.ndarray,
                    S: np.ndarray, C: np.ndarray):
    valid = ~np.isnan(x)
    pv = np.outer(valid, valid).astype(float)

    xi = np.where(valid, x, 0.0)
    new_n = non_nan + pv
    safe_n = np.where(new_n > 0, new_n, 1.0)

    delta1 = pv * (xi[:, None] - mean)
    new_mean = mean + delta1 / safe_n
    delta2 = pv * (xi[:, None] - new_mean)

    new_S = S + pv * delta1 * delta2
    new_C = C + pv * delta1 * delta2.T
    return new_n, new_mean, new_S, new_C


def update_nan(n_total: int, non_nan: np.ndarray, mean: np.ndarray,
               S: np.ndarray, C: np.ndarray, sample: np.ndarray):
    arr = np.atleast_2d(np.asarray(sample, dtype=np.float64))
    cur_nn, cur_mean, cur_S, cur_C = non_nan, mean, S, C
    for row in arr:
        cur_nn, cur_mean, cur_S, cur_C = _update_row_nan(row, cur_nn, cur_mean, cur_S, cur_C)
    return n_total + len(arr), cur_nn, cur_mean, cur_S, cur_C


def _downdate_row_nan(x: np.ndarray, non_nan: np.ndarray, mean: np.ndarray,
                      S: np.ndarray, C: np.ndarray):
    valid = ~np.isnan(x)
    pv = np.outer(valid, valid).astype(float)

    xi = np.where(valid, x, 0.0)
    new_nn = non_nan - pv

    delta1 = pv * (xi[:, None] - mean)

    safe_new_n = np.where(new_nn > 0, new_nn, 1.0)
    new_mean = np.where(
        pv > 0,
        (non_nan * mean - xi[:, None]) / safe_new_n,
        mean
    )
    delta2 = pv * (xi[:, None] - new_mean)

    new_S = S - pv * delta1 * delta2
    new_C = C - pv * delta1 * delta2.T
    return new_nn, new_mean, new_S, new_C


def downdate_nan(n_total: int, non_nan: np.ndarray, mean: np.ndarray,
                 S: np.ndarray, C: np.ndarray, sample: np.ndarray):
    arr = np.atleast_2d(np.asarray(sample, dtype=np.float64))
    cur_nn, cur_mean, cur_S, cur_C = non_nan, mean, S, C
    for row in arr:
        cur_nn, cur_mean, cur_S, cur_C = _downdate_row_nan(row, cur_nn, cur_mean, cur_S, cur_C)
    if np.any(np.diag(cur_S) < -1e-10):
        print("Warning: downdate produced negative diagonal in S.")
    return n_total - len(arr), cur_nn, cur_mean, cur_S, cur_C


def corrcoef_from_state_nan(non_nan: np.ndarray, mean: np.ndarray,
                            S: np.ndarray, C: np.ndarray) -> np.ndarray:
    var = S * S.T
    std = np.sqrt(np.maximum(var, 0.0))
    corr = np.where(std > 0, C / std, 0.0)
    corr = np.nan_to_num(corr)
    return np.clip(corr, -1.0, 1.0)


def aggregate_pcc(data, exclude_sample=None, include_sample=None, state=None):
    """Pearson correlation — full recomputation every call."""
    if exclude_sample is not None:
        values = data.drop(exclude_sample.index)
    elif include_sample is not None:
        values = pd.concat([data, include_sample], axis=0, ignore_index=False)
    else:
        values = data

    if isinstance(values, (pd.DataFrame, pd.Series)):
        values = values.to_numpy(dtype=float).T
    elif isinstance(values, np.ndarray):
        values = values.T

    if np.isnan(values).any():
        raise ValueError("NaN values detected. Run with imputation option.")

    return np.corrcoef(values), None


def aggregate_napy_pcc(data, exclude_sample=None, include_sample=None, state=None):
    """NaN-aware Pearson correlation via napy — full recomputation every call."""
    if exclude_sample is not None:
        values = data.drop(exclude_sample.index)
    elif include_sample is not None:
        values = pd.concat([data, include_sample], axis=0, ignore_index=False)
    else:
        values = data

    if isinstance(values, (pd.DataFrame, pd.Series)):
        arr = values.to_numpy()
    elif isinstance(values, np.ndarray):
        arr = values

    arr = np.where(np.isnan(arr), -999.0, arr)

    res = napy.pearsonr(
        arr,
        nan_value=-999.0,
        axis=1,
        threads=1,
        return_types=['r2']
    )

    R2 = res['r2']
    if isinstance(R2, pd.DataFrame):
        R2 = R2.to_numpy()

    return R2, None


def aggregate_pcc_welford(data, exclude_sample=None, include_sample=None, state=None):
    """Pearson correlation via Welford online algorithm."""
    if state is None:
        n, mean, M2 = init_welford(data)
        state = WelfordState(n, mean, M2)
    elif exclude_sample is not None:
        n, mean, M2 = downdate(
            state.n, state.mean, state.M2, exclude_sample
        )
    elif include_sample is not None:
        n, mean, M2 = update(
            state.n, state.mean, state.M2, include_sample
        )

    corr = corrcoef_from_state(n, mean, M2)
    return corr, state


def aggregate_napy_pcc_welford(data, exclude_sample=None, include_sample=None, state=None):
    """NaN-aware Pearson correlation via pairwise Welford online algorithm."""
    if state is None:
        n_total, non_nan, mean, S, C = init_welford_nan(data)
        state = WelfordNaNState(n_total, non_nan, mean, S, C)
    elif exclude_sample is not None:
        n_total, non_nan, mean, S, C = downdate_nan(
            state.n_total, state.non_nan, state.mean, state.S, state.C, exclude_sample
        )
    elif include_sample is not None:
        n_total, non_nan, mean, S, C = update_nan(
            state.n_total, state.non_nan, state.mean, state.S, state.C, include_sample
        )

    corr = corrcoef_from_state_nan(non_nan, mean, S, C)
    return corr, state


class Aggregator:
    """
    Usage:
        agg = Aggregator(aggregate_fn, base_data)
        corr = agg.corrcoef()                           # base correlation
        corr = agg.corrcoef_with_sample(new_sample)     # base + sample
        corr = agg.corrcoef_without_sample(old_sample)  # base - sample
    """

    _welford_fns = {aggregate_pcc_welford, aggregate_napy_pcc_welford}

    def __init__(self, aggregate_fn, base_data):
        self.fn = aggregate_fn
        self.is_welford = aggregate_fn in self._welford_fns
        self.data = base_data

        if self.is_welford:
            self.corr, self.state = self.fn(np.asarray(base_data, dtype=np.float64))
        else:
            self.corr, _ = self.fn(self.data)
            self.state = None

    def corrcoef(self) -> np.ndarray:
        return self.corr

    def corrcoef_with_sample(self, sample) -> np.ndarray:
        if self.is_welford:
            corr, _ = self.fn(None, include_sample=sample, state=self._copy_state())
        else:
            corr, _ = self.fn(self.data, include_sample=sample)
        return corr

    def corrcoef_without_sample(self, sample) -> np.ndarray:
        if self.is_welford:
            corr, _ = self.fn(None, exclude_sample=sample, state=self._copy_state())
        else:
            corr, _ = self.fn(self.data, exclude_sample=sample)
        return corr

    def _copy_state(self):
        if isinstance(self.state, WelfordState):
            return WelfordState(self.state.n, self.state.mean.copy(), self.state.M2.copy())
        elif isinstance(self.state, WelfordNaNState):
            return WelfordNaNState(
                self.state.n_total, self.state.non_nan.copy(),
                self.state.mean.copy(), self.state.S.copy(), self.state.C.copy(),
            )
        return None
    

def aggregate_spearman(data, exclude_sample=None, include_sample=None, state=None):
    """Spearman correlation via ranking + Pearson."""
    if exclude_sample is not None:
        values = data.drop(exclude_sample.index)
    elif include_sample is not None:
        values = pd.concat([data, include_sample], axis=0, ignore_index=False)
    else:
        values = data

    if isinstance(values, (pd.DataFrame, pd.Series)):
        values = values.to_numpy(dtype=float)
    elif isinstance(values, np.ndarray):
        values = values

    # Rank along samples (rows)
    ranks = np.apply_along_axis(
        lambda x: pd.Series(x).rank(method="average").to_numpy(),
        axis=0,
        arr=values
    )
    return np.corrcoef(ranks.T), None

def aggregate_napy_spearman(data, exclude_sample=None, include_sample=None, state=None):
    """NaN-aware Spearman correlation via napy — full recomputation every call."""
    
    if exclude_sample is not None:
        values = data.drop(exclude_sample.index)
    elif include_sample is not None:
        values = pd.concat([data, include_sample], axis=0, ignore_index=False)
    else:
        values = data

    if isinstance(values, (pd.DataFrame, pd.Series)):
        arr = values.to_numpy()
    elif isinstance(values, np.ndarray):
        arr = values

    # Replace NaNs with sentinel (same as Pearson version)
    arr = np.where(np.isnan(arr), -999.0, arr)

    res = napy.spearmanr(
        arr,
        nan_value=-999.0,
        axis=1,
        threads=1,
        return_types=['rho'] # Spearman coefficient
    )

    rho = res['rho']
    if isinstance(rho, pd.DataFrame):
        rho = rho.to_numpy()

    return rho, None