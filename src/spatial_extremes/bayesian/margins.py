"""Bayesian spatio-temporal hierarchical models for GEV margins.

The model in this module captures marginal parameter variation through
Gaussian processes, but assumes observations are conditionally independent.
It does not model residual extremal dependence between sites.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np


@dataclass
class BayesianGEVResult:
    """Fitted spatio-temporal GEV model and posterior samples."""

    model: Any
    inference_data: Any
    coordinates: np.ndarray
    time_values: np.ndarray
    covariate_mean: np.ndarray
    covariate_scale: np.ndarray

    def posterior_return_levels(
        self, return_periods: float | np.ndarray | list[float], time_index: int = -1
    ) -> np.ndarray:
        """Return posterior return-level draws for one time and all sites.

        A scalar return period returns an array shaped
        ``(chain * draw, site)``. An array of return periods returns
        ``(chain * draw, return_period, site)``.
        """
        periods = np.asarray(return_periods, dtype=float)
        scalar_period = periods.ndim == 0
        if periods.ndim > 1 or not np.all(np.isfinite(periods)) or np.any(periods <= 1):
            raise ValueError("return_periods must be finite values greater than 1")

        posterior = self.inference_data.posterior
        locations = np.asarray(posterior["location"].values, dtype=float)
        n_chains, n_draws, n_times, n_sites = locations.shape
        if isinstance(time_index, bool) or not isinstance(
            time_index, (int, np.integer)
        ):
            raise TypeError("time_index must be an integer")
        if not -n_times <= time_index < n_times:
            raise IndexError(f"time_index must be in [{-n_times}, {n_times - 1}]")
        locations = locations.reshape(n_chains * n_draws, n_times, n_sites)
        location = locations[:, time_index, :]
        scale = np.asarray(posterior["scale"].values, dtype=float).reshape(
            n_chains * n_draws, n_sites
        )
        shape = np.asarray(posterior["shape"].values, dtype=float).reshape(
            n_chains * n_draws, n_sites
        )

        periods = np.atleast_1d(periods)
        log_tail = np.log(-np.log1p(-1.0 / periods))
        shape_period = shape[:, None, :]
        gev_increment = np.expm1(-shape_period * log_tail[None, :, None])
        near_zero = np.abs(shape_period) < 1e-7
        safe_shape = np.where(near_zero, 1.0, shape_period)
        increments = np.where(
            near_zero,
            -log_tail[None, :, None],
            gev_increment / safe_shape,
        )
        result = location[:, None, :] + scale[:, None, :] * increments
        return result[:, 0, :] if scalar_period else result


def _validate_inputs(observations, coordinates, times, covariates):
    values = np.asarray(observations, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    if values.ndim != 2:
        raise ValueError("observations must have shape (time, site)")
    n_times, n_sites = values.shape
    if n_times < 2 or n_sites < 2:
        raise ValueError("At least two times and two sites are required")
    if coords.shape != (n_sites, 2) or not np.all(np.isfinite(coords)):
        raise ValueError(
            f"coordinates must be a finite array with shape ({n_sites}, 2)"
        )
    if np.isinf(values).any():
        raise ValueError("observations must be finite or missing (NaN)")
    if not np.isfinite(values).any():
        raise ValueError("observations must contain at least one finite value")
    if np.any(np.sum(np.isfinite(values), axis=0) == 0):
        raise ValueError("each site must have at least one finite observation")

    if times is None:
        time_values = np.arange(n_times, dtype=float)
    else:
        time_values = np.asarray(times, dtype=float)
        if time_values.shape != (n_times,) or not np.all(np.isfinite(time_values)):
            raise ValueError(f"times must be a finite array with shape ({n_times},)")
        differences = np.diff(time_values)
        if np.any(differences <= 0):
            raise ValueError("times must be strictly increasing")
        if not np.allclose(differences, differences[0], rtol=1e-6, atol=1e-10):
            raise ValueError("times must be equally spaced for the AR(1) model")

    if covariates is None:
        covariate_values = np.empty((n_times, 0), dtype=float)
        covariate_mean = np.empty(0, dtype=float)
        covariate_scale = np.empty(0, dtype=float)
    else:
        covariate_values = np.asarray(covariates, dtype=float)
        if (
            covariate_values.ndim != 2
            or covariate_values.shape[0] != n_times
            or covariate_values.shape[1] == 0
            or not np.all(np.isfinite(covariate_values))
        ):
            raise ValueError(
                "covariates must be a finite array with shape (time, covariate)"
            )
        covariate_mean = np.mean(covariate_values, axis=0)
        covariate_scale = np.std(covariate_values, axis=0)
        if np.any(covariate_scale == 0):
            raise ValueError("covariates must not contain constant columns")
        covariate_values = (covariate_values - covariate_mean) / covariate_scale
    return (
        values,
        coords,
        time_values,
        covariate_values,
        covariate_mean,
        covariate_scale,
    )


def fit_spatiotemporal_gev(
    observations,
    coordinates: np.ndarray,
    *,
    times: np.ndarray | None = None,
    covariates: np.ndarray | None = None,
    draws: int = 1_000,
    tune: int = 1_000,
    chains: int = 4,
    cores: int | None = None,
    target_accept: float = 0.9,
    random_seed: int | list[int] | None = None,
    sample_kwargs: dict | None = None,
) -> BayesianGEVResult:
    """Fit a Bayesian spatio-temporal hierarchical GEV margin model.

    Observations have shape ``(time, site)`` and may contain NaNs. Coordinates
    have shape ``(site, 2)`` and should be projected into suitable planar
    distance units. The location is the sum of a global intercept, optional
    standardized time-varying covariates, and a spatial Gaussian field whose
    dependence over time follows a stationary AR(1). Log-scale has a static
    spatial Gaussian field. Shape has a strongly regularized spatial field.
    The three spatial fields use separate Matérn-3/2 covariance parameters.

    PyMC is optional. Install it with ``pip install spatial-extremes[bayesian]``.
    The returned posterior includes pointwise GEV marginal inference and
    ``posterior_return_levels`` for return-level maps. The model assumes
    conditional independence across observations and does not represent
    residual extremal dependence.
    """
    values, coords, time_values, x, x_mean, x_scale = _validate_inputs(
        observations, coordinates, times, covariates
    )
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if isinstance(tune, bool) or not isinstance(tune, int) or tune < 1:
        raise ValueError("tune must be a positive integer")
    if isinstance(chains, bool) or not isinstance(chains, int) or chains < 1:
        raise ValueError("chains must be a positive integer")
    if not np.isfinite(target_accept) or not 0.5 < target_accept < 1:
        raise ValueError("target_accept must be between 0.5 and 1")
    if sample_kwargs is not None and not isinstance(sample_kwargs, dict):
        raise TypeError("sample_kwargs must be a dictionary")

    warnings.warn(
        "This hierarchical GEV model estimates marginal distributions but "
        "assumes conditional independence across sites and times. It does not "
        "model residual extremal dependence, so uncertainty in joint hazard "
        "quantities (such as simultaneous flooding) is likely understated. "
        "Do not use this model by itself for joint hazard probabilities.",
        UserWarning,
        stacklevel=2,
    )

    try:
        pm: Any = import_module("pymc")
        pt: Any = import_module("pytensor.tensor")
    except ImportError as exc:
        raise ImportError(
            "fit_spatiotemporal_gev requires the optional PyMC dependency; "
            "install it with `pip install spatial-extremes[bayesian]`"
        ) from exc

    n_times, n_sites = values.shape
    finite = np.isfinite(values)
    obs_times, obs_sites = np.nonzero(finite)
    observed = values[finite]

    coord_center = np.mean(coords, axis=0)
    coord_scale = np.sqrt(np.mean(np.sum((coords - coord_center) ** 2, axis=1)))
    if coord_scale == 0:
        coord_scale = 1.0
    normalized_coords = (coords - coord_center) / coord_scale
    data_scale = max(float(np.std(observed)), 1e-3)
    data_location = float(np.mean(observed))
    cov_names = [f"covariate_{i}" for i in range(x.shape[1])]

    model_coords = {
        "time": time_values,
        "site": np.arange(n_sites),
        "xy": np.arange(2),
    }
    if cov_names:
        model_coords["covariate"] = cov_names

    with pm.Model(coords=model_coords) as model:
        coord_data = pm.Data("coordinates", normalized_coords, dims=("site", "xy"))

        location_intercept = pm.Normal(
            "location_intercept", mu=data_location, sigma=2 * data_scale
        )
        if cov_names:
            beta = pm.Normal(
                "covariate_effect", mu=0, sigma=data_scale, dims="covariate"
            )
            trend = pt.dot(x, beta)
        else:
            trend = pt.zeros((n_times,))

        loc_amplitude = pm.HalfNormal("location_gp_amplitude", sigma=data_scale)
        loc_length = pm.LogNormal("location_gp_length_scale", mu=0, sigma=0.7)
        loc_phi = pm.Uniform("location_ar1", lower=-0.95, upper=0.95)
        loc_covariance = (
            pm.gp.cov.Matern32(input_dim=2, ls=loc_length)(coord_data)
            * loc_amplitude**2
            + pt.eye(n_sites) * 1e-6
        )
        loc_cholesky = pt.linalg.cholesky(loc_covariance)
        loc_white_noise = pm.Normal(
            "location_white_noise", mu=0, sigma=1, dims=("time", "site")
        )
        loc_spatial_innovations = pt.dot(loc_white_noise, loc_cholesky.T)
        loc_fields = [loc_spatial_innovations[0]]
        innovation_scale = pt.sqrt(1 - loc_phi**2)
        for time_index in range(1, n_times):
            loc_fields.append(
                loc_phi * loc_fields[-1]
                + innovation_scale * loc_spatial_innovations[time_index]
            )
        location = location_intercept + trend[:, None] + pt.stack(loc_fields, axis=0)

        log_scale_intercept = pm.Normal(
            "log_scale_intercept", mu=np.log(data_scale), sigma=1.0
        )
        scale_amplitude = pm.HalfNormal("log_scale_gp_amplitude", sigma=0.5)
        scale_length = pm.LogNormal("log_scale_gp_length_scale", mu=0, sigma=0.7)
        scale_covariance = (
            pm.gp.cov.Matern32(input_dim=2, ls=scale_length)(coord_data)
            + pt.eye(n_sites) * 1e-6
        )
        scale_cholesky = pt.linalg.cholesky(scale_covariance)
        scale_white_noise = pm.Normal(
            "log_scale_white_noise", mu=0, sigma=1, dims="site"
        )
        log_scale = log_scale_intercept + scale_amplitude * pt.dot(
            scale_white_noise, scale_cholesky.T
        )
        scale = pm.Deterministic("scale", pt.exp(log_scale), dims="site")

        shape_intercept = pm.Normal("shape_intercept", mu=0, sigma=0.1)
        shape_amplitude = pm.HalfNormal("shape_gp_amplitude", sigma=0.05)
        shape_length = pm.LogNormal("shape_gp_length_scale", mu=0, sigma=0.7)
        shape_covariance = (
            pm.gp.cov.Matern32(input_dim=2, ls=shape_length)(coord_data)
            + pt.eye(n_sites) * 1e-6
        )
        shape_cholesky = pt.linalg.cholesky(shape_covariance)
        shape_white_noise = pm.Normal("shape_white_noise", mu=0, sigma=1, dims="site")
        shape = pm.Deterministic(
            "shape",
            shape_intercept
            + shape_amplitude * pt.dot(shape_white_noise, shape_cholesky.T),
            dims="site",
        )
        location = pm.Deterministic("location", location, dims=("time", "site"))

        mu_obs = location[obs_times, obs_sites]
        sigma_obs = scale[obs_sites]
        xi_obs = shape[obs_sites]
        standardized = (observed - mu_obs) / sigma_obs
        support = 1 + xi_obs * standardized
        small_shape = pt.abs(xi_obs) < 1e-6
        safe_shape = pt.where(small_shape, 1.0, xi_obs)
        safe_support = pt.maximum(1 + safe_shape * standardized, 1e-12)
        gev_logp = (
            -pt.log(sigma_obs)
            - (1 + 1 / safe_shape) * pt.log(safe_support)
            - safe_support ** (-1 / safe_shape)
        )
        gumbel_logp = -pt.log(sigma_obs) - standardized - pt.exp(-standardized)
        log_likelihood = pt.where(small_shape, gumbel_logp, gev_logp)
        log_likelihood = pt.where(support > 0, log_likelihood, -np.inf)
        pm.Potential("gev_log_likelihood", pt.sum(log_likelihood))

        sampling_options = dict(sample_kwargs or {})
        reserved = {
            "draws",
            "tune",
            "chains",
            "cores",
            "target_accept",
            "random_seed",
            "return_inferencedata",
        }
        collision = reserved & sampling_options.keys()
        if collision:
            raise ValueError(
                f"sample_kwargs cannot override explicit arguments: {sorted(collision)}"
            )
        inference_data = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            **sampling_options,
        )

    return BayesianGEVResult(
        model=model,
        inference_data=inference_data,
        coordinates=coords.copy(),
        time_values=time_values.copy(),
        covariate_mean=x_mean,
        covariate_scale=x_scale,
    )
