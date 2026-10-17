"""
Pairwise (composite) likelihood fitting.

The full joint density of a max-stable process is generally difficult
to evaluate for many sites. This module estimates dependence parameters
by maximizing the sum of bivariate log-densities over site pairs and
replicates, a composite-likelihood approach used in the spatial-extremes
literature (Padoan, Ribatet & Sisson 2010).

Works with any model exposing:
  - `bivariate_logpdf(z1, z2, h)` (vectorized over replicates)
  - `from_free_params(free_params)` classmethod
  - `n_free_params` class attribute

i.e. SmithModel, SchlatherModel, BrownResnickModel all plug in directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import optimize

try:
    import polars as pl
except ImportError:
    pl = None


def _to_numpy(df):
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_numpy()
    if isinstance(df, np.ndarray):
        return df
    return df.to_numpy()


@dataclass
class PairwiseFitResult:
    model: object
    free_params: np.ndarray
    nllh: float
    converged: bool
    n_pairs_used: int
    n_replicates: int
    optimizer_result: object


def fit_pairwise_likelihood(
    model_class,
    frechet_df,
    coords: np.ndarray,
    x0_free: np.ndarray | None = None,
    max_pairs: int | None = None,
    method: str = "Nelder-Mead",
    rng: np.random.Generator | None = None,
) -> PairwiseFitResult:
    """Fit a max-stable model by maximizing the pairwise log-likelihood.

    Parameters
    ----------
    model_class : SmithModel, SchlatherModel, or BrownResnickModel (the
        class itself, not an instance -- it's constructed internally at
        each optimizer step via `model_class.from_free_params(...)`).
    frechet_df : pandas or polars DataFrame, or 2-D NumPy array. Columns
        are sites with unit-Frechet margins; rows are independent
        replicates. Missing values are allowed and each pair uses only
        rows observed at both sites.
    coords : (D, 2) array_like of site coordinates, same order as columns.
    x0_free : optional starting point in free-parameter space; defaults
        to a vector of zeros (which is a reasonable "unit scale, no
        anisotropy" starting guess under each model's reparametrization).
    max_pairs : if the number of sites is large, randomly subsample this
        many site pairs per optimizer evaluation (fixed once, not
        resampled per iteration) to keep fitting tractable. `None` uses
        all pairs.

    Notes
    -----
    The method assumes independent replicates from a stationary spatial
    model and ignorable missingness. The fit does not estimate marginal
    parameters or provide composite-likelihood uncertainty estimates.
    A successful optimizer termination does not establish model adequacy.
    """
    values = np.asarray(_to_numpy(frechet_df), dtype=float)
    if values.ndim != 2:
        raise ValueError("frechet_df must be a two-dimensional table of observations")
    n_obs, D = values.shape
    coords = np.asarray(coords, dtype=float)
    if D < 2:
        raise ValueError("At least two sites are required for pairwise fitting")
    if coords.shape != (D, 2):
        raise ValueError(f"coords must have shape ({D}, 2), got {coords.shape}")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite values")
    if np.isinf(values).any():
        raise ValueError("Observations must be finite or missing (NaN)")
    observed = np.isfinite(values)
    if np.any(values[observed] <= 0):
        raise ValueError("Unit-Frechet observations must be strictly positive")

    if max_pairs is not None and (
        isinstance(max_pairs, bool) or not isinstance(max_pairs, (int, np.integer)) or max_pairs < 1
    ):
        raise ValueError("max_pairs must be a positive integer")

    pairs = []
    for i, j in combinations(range(D), 2):
        valid = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
        if np.any(valid):
            pairs.append((i, j, values[valid, i], values[valid, j]))
    if not pairs:
        raise ValueError("No site pairs have jointly observed values")

    if max_pairs is not None and max_pairs < len(pairs):
        rng = rng or np.random.default_rng()
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]

    if x0_free is None:
        x0_free = np.zeros(model_class.n_free_params)

    def neg_composite_llh(free_params):
        try:
            model = model_class.from_free_params(free_params)
        except (ValueError, np.linalg.LinAlgError):
            return 1e12
        total = 0.0
        for i, j, z1, z2 in pairs:
            h = coords[i] - coords[j]
            logpdf = model.bivariate_logpdf(z1, z2, h)
            if not np.all(np.isfinite(logpdf)):
                return 1e12
            total += np.sum(logpdf)
        return -total

    result = optimize.minimize(
        neg_composite_llh, x0=np.asarray(x0_free, dtype=float), method=method,
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6}
        if method == "Nelder-Mead" else {"maxiter": 5000},
    )

    fitted_model = model_class.from_free_params(result.x)
    return PairwiseFitResult(
        model=fitted_model, free_params=result.x, nllh=result.fun,
        converged=bool(result.success), n_pairs_used=len(pairs),
        n_replicates=n_obs, optimizer_result=result,
    )
