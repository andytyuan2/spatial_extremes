"""
Pairwise (composite) likelihood fitting and Bayesian inference.

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

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import polars as pl
from scipy import optimize


def _to_numpy(df):
    if isinstance(df, pl.DataFrame):
        return df.to_numpy()
    if isinstance(df, np.ndarray):
        return df
    if type(df).__module__.split(".", maxsplit=1)[0] == "pandas":
        return df.to_numpy()
    raise TypeError(
        "frechet_df must be a Polars DataFrame, pandas DataFrame, or NumPy array"
    )


@dataclass
class PairwiseFitResult:
    model: object
    free_params: np.ndarray
    nllh: float
    converged: bool
    n_pairs_used: int
    n_replicates: int
    optimizer_result: object


@dataclass
class PairwiseBayesResult:
    """Samples and curvature diagnostics for a pairwise-likelihood posterior."""

    free_param_samples: np.ndarray
    log_posterior_samples: np.ndarray
    acceptance_rate: float
    pairwise_fit: PairwiseFitResult
    sensitivity_matrix: np.ndarray
    variability_matrix: np.ndarray
    curvature_matrix: np.ndarray
    curvature_adjustment: np.ndarray
    model_class: type

    @property
    def posterior_mean_free_params(self) -> np.ndarray:
        return np.mean(self.free_param_samples, axis=0)

    @property
    def posterior_mean_model(self):
        """Model instantiated at the posterior mean in free-parameter space."""
        return self.model_class.from_free_params(self.posterior_mean_free_params)


def _prepare_pairwise_data(
    frechet_df,
    coords: np.ndarray,
    max_pairs: int | None,
    rng: np.random.Generator | None,
):
    values = np.asarray(_to_numpy(frechet_df), dtype=float)
    if values.ndim != 2:
        raise ValueError("frechet_df must be a two-dimensional table of observations")
    _, D = values.shape
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
        isinstance(max_pairs, bool)
        or not isinstance(max_pairs, (int, np.integer))
        or max_pairs < 1
    ):
        raise ValueError("max_pairs must be a positive integer")

    pairs = []
    for i, j in combinations(range(D), 2):
        rows = np.flatnonzero(np.isfinite(values[:, i]) & np.isfinite(values[:, j]))
        if rows.size:
            pairs.append((i, j, rows, values[rows, i], values[rows, j]))
    if not pairs:
        raise ValueError("No site pairs have jointly observed values")

    if max_pairs is not None and max_pairs < len(pairs):
        rng = rng or np.random.default_rng()
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]
    return values, coords, pairs


def _replicate_loglikelihoods(model_class, free_params, coords, pairs, n_obs):
    """Return composite log-likelihood contributions, summed by replicate."""
    try:
        model = model_class.from_free_params(free_params)
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return None

    contributions = np.zeros(n_obs, dtype=float)
    for i, j, rows, z1, z2 in pairs:
        logpdf = np.asarray(
            model.bivariate_logpdf(z1, z2, coords[i] - coords[j]), dtype=float
        )
        if logpdf.shape != rows.shape or not np.all(np.isfinite(logpdf)):
            return None
        contributions[rows] += logpdf
    return contributions


def _symmetric_positive_definite_sqrt(matrix, *, inverse=False):
    matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    threshold = np.max(np.abs(eigenvalues)) * 1e-10
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= threshold:
        raise ValueError(
            "Curvature adjustment requires positive-definite information "
            "matrices; the pairwise fit may be weakly identified or the "
            "number of independent replicates may be insufficient"
        )
    powers = eigenvalues ** (-0.5 if inverse else 0.5)
    return (eigenvectors * powers) @ eigenvectors.T


def _curvature_adjustment(model_class, mle, coords, pairs, n_obs):
    """Estimate the Godambe curvature and its local affine adjustment."""
    mle = np.asarray(mle, dtype=float)
    n_params = mle.size
    steps = 1e-4 * np.maximum(1.0, np.abs(mle))

    def contributions(params):
        result = _replicate_loglikelihoods(model_class, params, coords, pairs, n_obs)
        if result is None:
            raise ValueError("The pairwise log-likelihood is invalid near its optimum")
        return result

    center = contributions(mle)
    scores = np.empty((n_obs, n_params), dtype=float)
    for k, step in enumerate(steps):
        offset = np.zeros(n_params)
        offset[k] = step
        scores[:, k] = (contributions(mle + offset) - contributions(mle - offset)) / (
            2 * step
        )

    # Replicates are the independent sampling units; each score first
    # aggregates contributions across all pairs within that replicate.
    variability = scores.T @ scores

    sensitivity = np.empty((n_params, n_params), dtype=float)
    for i, step_i in enumerate(steps):
        ei = np.zeros(n_params)
        ei[i] = step_i
        sensitivity[i, i] = (
            -(
                np.sum(contributions(mle + ei))
                - 2 * np.sum(center)
                + np.sum(contributions(mle - ei))
            )
            / step_i**2
        )
        for j in range(i):
            ej = np.zeros(n_params)
            ej[j] = steps[j]
            second_derivative = (
                np.sum(contributions(mle + ei + ej))
                - np.sum(contributions(mle + ei - ej))
                - np.sum(contributions(mle - ei + ej))
                + np.sum(contributions(mle - ei - ej))
            ) / (4 * step_i * steps[j])
            sensitivity[i, j] = sensitivity[j, i] = -second_derivative
    sensitivity = (sensitivity + sensitivity.T) / 2
    variability = (variability + variability.T) / 2

    sensitivity_inv_sqrt = _symmetric_positive_definite_sqrt(sensitivity, inverse=True)
    variability_inv_sqrt = _symmetric_positive_definite_sqrt(variability, inverse=True)
    del variability_inv_sqrt  # validates that the score variability is full rank
    curvature = sensitivity @ np.linalg.solve(variability, sensitivity)
    curvature = (curvature + curvature.T) / 2
    curvature_sqrt = _symmetric_positive_definite_sqrt(curvature)

    # For ell_adj(theta) = ell(theta_hat + C(theta-theta_hat)),
    # C.T @ H @ C equals the Godambe information H J^-1 H.
    adjustment = sensitivity_inv_sqrt @ curvature_sqrt
    return sensitivity, variability, curvature, adjustment


def _fit_pairwise_prepared(
    model_class, n_obs, coords, pairs, x0_free=None, method="Nelder-Mead"
):
    if x0_free is None:
        x0_free = np.zeros(model_class.n_free_params)

    def neg_composite_llh(free_params):
        try:
            model = model_class.from_free_params(free_params)
        except (ValueError, np.linalg.LinAlgError):
            return 1e12
        total = 0.0
        for i, j, _, z1, z2 in pairs:
            logpdf = model.bivariate_logpdf(z1, z2, coords[i] - coords[j])
            if not np.all(np.isfinite(logpdf)):
                return 1e12
            total += np.sum(logpdf)
        return -total

    result = optimize.minimize(
        neg_composite_llh,
        x0=np.asarray(x0_free, dtype=float),
        method=method,
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6}
        if method == "Nelder-Mead"
        else {"maxiter": 5000},
    )
    fitted_model = model_class.from_free_params(result.x)
    return PairwiseFitResult(
        model=fitted_model,
        free_params=result.x,
        nllh=result.fun,
        converged=bool(result.success),
        n_pairs_used=len(pairs),
        n_replicates=n_obs,
        optimizer_result=result,
    )


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
    frechet_df : Polars DataFrame, 2-D NumPy array, or (with the ``pandas``
        extra installed) pandas DataFrame. Columns are sites with
        unit-Frechet margins; rows are independent replicates. Missing
        values are allowed and each pair uses only rows observed at both
        sites.
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
    values, coords, pairs = _prepare_pairwise_data(frechet_df, coords, max_pairs, rng)
    n_obs = values.shape[0]

    return _fit_pairwise_prepared(
        model_class, n_obs, coords, pairs, x0_free=x0_free, method=method
    )


def sample_pairwise_likelihood_posterior(
    model_class,
    frechet_df,
    coords: np.ndarray,
    n_samples: int = 2_000,
    burn_in: int = 1_000,
    thin: int = 1,
    log_prior: Callable[[np.ndarray], float] | None = None,
    prior_scale: float = 2.5,
    proposal_scale: float | None = None,
    x0_free: np.ndarray | None = None,
    max_pairs: int | None = None,
    rng: np.random.Generator | None = None,
) -> PairwiseBayesResult:
    """Sample a curvature-adjusted Bayesian pairwise-likelihood posterior.

    The adjustment is local to the pairwise maximum-likelihood estimate. If
    ``ell`` is the pairwise log-likelihood, ``H`` its observed negative
    Hessian, and ``J`` the sum of outer products of replicate-level scores,
    the affine parameter map is chosen so the adjusted curvature is
    ``H @ inv(J) @ H``. This gives the adjusted likelihood the Godambe
    curvature used to correct the usual composite-likelihood posterior
    concentration (Ribatet, Cooley & Davison, 2012).

    Parameters
    ----------
    model_class : model class
        A class accepted by :func:`fit_pairwise_likelihood`.
    frechet_df, coords : array-like
        Unit-Frechet observations and their two-dimensional site coordinates,
        as for :func:`fit_pairwise_likelihood`.
    n_samples : int
        Number of retained Metropolis samples.
    burn_in : int
        Number of initial Metropolis iterations to discard.
    thin : int
        Retain one sample every ``thin`` iterations after burn-in.
    log_prior : callable, optional
        Function returning the log prior density for a vector of free
        parameters. It may return ``-inf`` outside its support. By default,
        independent zero-centered normal priors with standard deviation
        ``prior_scale`` are used on the free parameters.
    prior_scale : float
        Standard deviation for the default normal prior.
    proposal_scale : float, optional
        Multiplier for the proposal standard deviation. By default, uses
        ``2.38 / sqrt(number of parameters)`` with the inverse Godambe
        information as proposal covariance.
    x0_free, max_pairs : optional
        Starting point and fixed site-pair subsample, as for
        :func:`fit_pairwise_likelihood`.
    rng : numpy.random.Generator, optional
        Random number generator used for pair subsampling and MCMC.

    Notes
    -----
    The returned ``posterior_mean_model`` is instantiated at the posterior
    mean in free-parameter space. The method assumes independent replicates.
    Curvature matrices are finite-difference estimates and must be positive
    definite; weakly identified fits are rejected rather than silently
    regularized.
    """
    warnings.warn(
        "Curvature-adjusted pairwise-likelihood inference is an approximate, "
        "local adjustment. It assumes independent replicates, treats "
        "unit-Frechet margins as known, and does not assess MCMC convergence; "
        "check the chain and sensitivity to the prior before interpreting "
        "posterior summaries.",
        UserWarning,
        stacklevel=2,
    )

    if (
        isinstance(n_samples, bool)
        or not isinstance(n_samples, (int, np.integer))
        or n_samples < 1
    ):
        raise ValueError("n_samples must be a positive integer")
    if (
        isinstance(burn_in, bool)
        or not isinstance(burn_in, (int, np.integer))
        or burn_in < 0
    ):
        raise ValueError("burn_in must be a nonnegative integer")
    if isinstance(thin, bool) or not isinstance(thin, (int, np.integer)) or thin < 1:
        raise ValueError("thin must be a positive integer")
    if not np.isfinite(prior_scale) or prior_scale <= 0:
        raise ValueError("prior_scale must be positive and finite")
    if proposal_scale is not None and (
        not np.isfinite(proposal_scale) or proposal_scale <= 0
    ):
        raise ValueError("proposal_scale must be positive and finite")

    rng = rng or np.random.default_rng()
    values, coords, pairs = _prepare_pairwise_data(frechet_df, coords, max_pairs, rng)
    fit = _fit_pairwise_prepared(
        model_class,
        values.shape[0],
        coords,
        pairs,
        x0_free=x0_free,
    )
    sensitivity, variability, curvature, adjustment = _curvature_adjustment(
        model_class, fit.free_params, coords, pairs, values.shape[0]
    )

    if log_prior is None:
        normalizer = fit.free_params.size * np.log(prior_scale * np.sqrt(2 * np.pi))

        def log_prior(params):
            return float(-0.5 * np.sum((params / prior_scale) ** 2) - normalizer)

    def evaluate(params):
        prior_value = float(log_prior(np.asarray(params, dtype=float)))
        if np.isnan(prior_value) or np.isposinf(prior_value):
            raise ValueError("log_prior must return a finite value or -inf")
        if not np.isfinite(prior_value):
            return -np.inf
        adjusted_params = fit.free_params + adjustment @ (params - fit.free_params)
        contributions = _replicate_loglikelihoods(
            model_class, adjusted_params, coords, pairs, values.shape[0]
        )
        if contributions is None:
            return -np.inf
        return float(np.sum(contributions) + prior_value)

    current = fit.free_params.copy()
    current_log_posterior = evaluate(current)
    if not np.isfinite(current_log_posterior):
        raise ValueError(
            "The log prior must be finite at the pairwise maximum-likelihood estimate"
        )
    covariance = np.linalg.inv(curvature)
    scale = proposal_scale or 2.38 / np.sqrt(current.size)
    proposal_cholesky = np.linalg.cholesky(covariance) * scale
    n_iterations = int(burn_in + n_samples * thin)
    samples = np.empty((n_samples, current.size), dtype=float)
    log_posterior_samples = np.empty(n_samples, dtype=float)
    accepted = 0
    retained = 0

    for iteration in range(1, n_iterations + 1):
        proposal = current + proposal_cholesky @ rng.standard_normal(current.size)
        proposed_log_posterior = evaluate(proposal)
        if np.log(rng.random()) < proposed_log_posterior - current_log_posterior:
            current = proposal
            current_log_posterior = proposed_log_posterior
            accepted += 1
        if iteration > burn_in and (iteration - burn_in) % thin == 0:
            samples[retained] = current
            log_posterior_samples[retained] = current_log_posterior
            retained += 1

    return PairwiseBayesResult(
        free_param_samples=samples,
        log_posterior_samples=log_posterior_samples,
        acceptance_rate=accepted / n_iterations,
        pairwise_fit=fit,
        sensitivity_matrix=sensitivity,
        variability_matrix=variability,
        curvature_matrix=curvature,
        curvature_adjustment=adjustment,
        model_class=model_class,
    )
