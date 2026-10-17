import numpy as np
import pandas as pd
import pytest
from scipy import stats

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
    df = pd.DataFrame({
        "site_a": stats.genextreme.rvs(c=0.0, loc=0.0, scale=1.0, size=500, random_state=rng),
        "site_b": stats.genextreme.rvs(c=-0.2, loc=5.0, scale=2.0, size=500, random_state=rng),
    })
    result = fit_gev_by_site(df)
    assert set(result["site"]) == {"site_a", "site_b"}
    assert (result["converged"]).all()


def test_fit_gev_by_site_long_format_with_missing_values():
    rng = np.random.default_rng(4)
    a = stats.gumbel_r.rvs(loc=1.0, scale=2.0, size=80, random_state=rng)
    b = stats.gumbel_r.rvs(loc=4.0, scale=1.5, size=80, random_state=rng)
    a[::10] = np.nan
    b[::8] = np.nan
    df = pd.DataFrame({
        "station_id": np.repeat(["a", "b"], 80),
        "year": np.tile(np.arange(80), 2),
        "annual_max": np.concatenate([a, b]),
        "elevation": np.repeat([100.0, 250.0], 80),
    })

    result = fit_gev_by_site(df, site_col="station_id", value_col="annual_max")

    assert result["site"].tolist() == ["a", "b"]
    assert result["n_obs"].tolist() == [72, 70]
    assert result["converged"].all()


def test_fit_gev_by_site_accepts_polars_long_table():
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(5)
    x = stats.gumbel_r.rvs(loc=2.0, scale=1.5, size=40, random_state=rng)
    x[::9] = np.nan
    df = pl.DataFrame({
        "station_id": ["site_a"] * len(x),
        "year": np.arange(len(x)),
        "annual_max": x,
    })

    result = fit_gev_by_site(df, site_col="station_id", value_col="annual_max")

    assert result.loc[0, "site"] == "site_a"
    assert result.loc[0, "n_obs"] == np.isfinite(x).sum()


def test_fit_gev_rejects_infinite_observations():
    x = np.linspace(0.0, 1.0, 20)
    x[0] = np.inf
    with pytest.raises(ValueError, match="finite or missing"):
        fit_gev(x)
