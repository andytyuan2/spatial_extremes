from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from scipy import stats

from spatial_extremes.bayesian.margins import (
    BayesianGEVResult,
    _validate_inputs,
    fit_spatiotemporal_gev,
)
from spatial_extremes.gev import fit_gev, fit_gev_by_site


def test_fit_gev_recovers_known_parameters():
    rng = np.random.default_rng(0)
    true_mu, true_sigma, true_xi = 10.0, 2.0, 0.1
    # scipy's genextreme uses c = -xi
    x = stats.genextreme.rvs(c=-true_xi, loc=true_mu, scale=true_sigma,
                              size=2000, random_state=rng)
    res = fit_gev(x)
    assert res.converged
    assert abs(res.loc - true_mu) < 0.3
    assert abs(res.scale - true_sigma) < 0.3
    assert abs(res.shape - true_xi) < 0.1


def test_fit_gev_gumbel_case():
    rng = np.random.default_rng(1)
    x = stats.gumbel_r.rvs(loc=5.0, scale=1.5, size=2000, random_state=rng)
    res = fit_gev(x)
    assert abs(res.shape) < 0.1  # should be close to 0 (Gumbel limit)


def test_to_unit_frechet_is_uniform_on_frechet_scale():
    rng = np.random.default_rng(2)
    x = stats.genextreme.rvs(c=0.0, loc=0.0, scale=1.0, size=5000, random_state=rng)
    res = fit_gev(x)
    z = res.to_unit_frechet(x)
    # unit Frechet CDF is exp(-1/z); transformed values should be ~Uniform(0,1)
    u = np.exp(-1.0 / z)
    _, pval = stats.kstest(u, "uniform")
    assert pval > 0.01


def test_fit_gev_by_site_multiple_columns():
    rng = np.random.default_rng(3)
    df = pl.DataFrame({
        "site_a": stats.genextreme.rvs(c=0.0, loc=0.0, scale=1.0, size=500, random_state=rng),
        "site_b": stats.genextreme.rvs(c=-0.2, loc=5.0, scale=2.0, size=500, random_state=rng),
    })
    result = fit_gev_by_site(df)
    assert isinstance(result, pl.DataFrame)
    assert set(result.get_column("site")) == {"site_a", "site_b"}
    assert result.get_column("converged").all()


def test_fit_gev_by_site_long_format_with_missing_values():
    rng = np.random.default_rng(4)
    a = stats.gumbel_r.rvs(loc=1.0, scale=2.0, size=80, random_state=rng)
    b = stats.gumbel_r.rvs(loc=4.0, scale=1.5, size=80, random_state=rng)
    a[::10] = np.nan
    b[::8] = np.nan
    df = pl.DataFrame({
        "station_id": np.repeat(["a", "b"], 80),
        "year": np.tile(np.arange(80), 2),
        "annual_max": np.concatenate([a, b]),
        "elevation": np.repeat([100.0, 250.0], 80),
    })

    result = fit_gev_by_site(df, site_col="station_id", value_col="annual_max")

    assert result.get_column("site").to_list() == ["a", "b"]
    assert result.get_column("n_obs").to_list() == [72, 70]
    assert result.get_column("converged").all()


def test_fit_gev_by_site_accepts_polars_long_table():
    rng = np.random.default_rng(5)
    x = stats.gumbel_r.rvs(loc=2.0, scale=1.5, size=40, random_state=rng)
    x[::9] = np.nan
    df = pl.DataFrame({
        "station_id": ["site_a"] * len(x),
        "year": np.arange(len(x)),
        "annual_max": x,
    })

    result = fit_gev_by_site(df, site_col="station_id", value_col="annual_max")

    assert result.item(0, "site") == "site_a"
    assert result.item(0, "n_obs") == np.isfinite(x).sum()


def test_fit_gev_by_site_accepts_legacy_pandas_and_returns_polars():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(6)
    df = pd.DataFrame({
        "site_a": stats.gumbel_r.rvs(loc=0.0, scale=1.0, size=50, random_state=rng),
    })

    result = fit_gev_by_site(df)

    assert isinstance(result, pl.DataFrame)
    assert result.item(0, "site") == "site_a"


def test_fit_gev_rejects_infinite_observations():
    x = np.linspace(0.0, 1.0, 20)
    x[0] = np.inf
    with pytest.raises(ValueError, match="finite or missing"):
        fit_gev(x)


def test_spatiotemporal_gev_rejects_irregular_times():
    observations = np.ones((4, 2))
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="equally spaced"):
        _validate_inputs(observations, coordinates, [1, 2, 4, 5], None)


def test_spatiotemporal_gev_rejects_constant_covariates():
    observations = np.ones((4, 2))
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="constant columns"):
        _validate_inputs(observations, coordinates, None, np.ones((4, 1)))


def test_bayesian_gev_return_levels_match_gumbel_quantiles():
    posterior = {
        "location": SimpleNamespace(values=np.zeros((1, 2, 3, 2))),
        "scale": SimpleNamespace(values=np.array([[[2.0, 3.0], [2.0, 3.0]]])),
        "shape": SimpleNamespace(values=np.zeros((1, 2, 2))),
    }
    result = BayesianGEVResult(
        model=None,
        inference_data=SimpleNamespace(posterior=posterior),
        coordinates=np.zeros((2, 2)),
        time_values=np.arange(3),
        covariate_mean=np.empty(0),
        covariate_scale=np.empty(0),
    )

    actual = result.posterior_return_levels(100, time_index=1)
    expected = np.array([2.0, 3.0]) * -np.log(-np.log1p(-1 / 100))
    assert actual.shape == (2, 2)
    assert actual == pytest.approx(np.tile(expected, (2, 1)))


def test_spatiotemporal_gev_samples_and_returns_marginal_levels():
    pytest.importorskip("pymc")
    rng = np.random.default_rng(7)
    observations = rng.gumbel(loc=3, scale=1.2, size=(6, 3))
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    with pytest.warns(UserWarning, match="does not model residual extremal dependence"):
        result = fit_spatiotemporal_gev(
            observations,
            coordinates,
            times=np.arange(2000, 2006),
            covariates=np.arange(6, dtype=float)[:, None],
            draws=8,
            tune=8,
            chains=1,
            cores=1,
            random_seed=8,
            sample_kwargs={
                "progressbar": False,
                "compute_convergence_checks": False,
            },
        )

    return_levels = result.posterior_return_levels([20, 50])
    assert return_levels.shape == (8, 2, 3)
    assert np.all(np.isfinite(return_levels))
    assert result.inference_data.posterior.coords["time"].values.tolist() == list(
        range(2000, 2006)
    )
    assert result.covariate_mean == pytest.approx([2.5])
