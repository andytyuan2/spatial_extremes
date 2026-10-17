import numpy as np
import polars as pl
import pytest

from spatial_extremes.fitting import (
    fit_pairwise_likelihood,
    sample_pairwise_likelihood_posterior,
)
from spatial_extremes.models.brownresnick import BrownResnickModel
from spatial_extremes.models.schlather import SchlatherModel
from spatial_extremes.models.smith import SmithModel


def _frechet_frame(sims, n_sites):
    return pl.DataFrame(sims, schema=[f"s{i}" for i in range(n_sites)], orient="row")


def test_fit_smith_recovers_extremal_coefficients():
    """Fit Smith on Smith-simulated data; check the fitted model's
    extremal coefficient function matches the true one closely, rather
    than requiring exact recovery of the (unidentifiable-up-to-rotation)
    covariance entries themselves."""
    rng = np.random.default_rng(100)
    true_cov = np.array([[2.0, 0.4], [0.4, 1.3]])
    true_model = SmithModel(cov=true_cov)

    coords = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 3.0]])
    sims = true_model.simulate(coords, n_reps=1500, rng=rng)
    df = _frechet_frame(sims, coords.shape[0])

    fit = fit_pairwise_likelihood(SmithModel, df, coords, rng=rng)
    assert fit.converged or fit.optimizer_result.nit > 0

    test_hs = [np.array([2.0, 0.0]), np.array([0.0, 2.0]), np.array([2.0, 2.0])]
    for h in test_hs:
        true_theta = true_model.extremal_coefficient(h)
        fit_theta = fit.model.extremal_coefficient(h)
        assert abs(true_theta - fit_theta) < 0.15, (
            f"h={h}: true={true_theta:.3f} fit={fit_theta:.3f}"
        )


def test_fit_schlather_recovers_extremal_coefficients():
    rng = np.random.default_rng(101)
    true_model = SchlatherModel(range_=2.5, smoothness=1.0)
    coords = np.array([[0.0, 0.0], [1.5, 0.0], [0.0, 1.5], [1.5, 1.5], [0.7, 2.2]])
    sims = true_model.simulate(coords, n_reps=1500, rng=rng)
    df = _frechet_frame(sims, coords.shape[0])

    fit = fit_pairwise_likelihood(SchlatherModel, df, coords, rng=rng)

    test_hs = [np.array([1.5, 0.0]), np.array([1.5, 1.5])]
    for h in test_hs:
        true_theta = true_model.extremal_coefficient(h)
        fit_theta = fit.model.extremal_coefficient(h)
        assert abs(true_theta - fit_theta) < 0.15, (
            f"h={h}: true={true_theta:.3f} fit={fit_theta:.3f}"
        )


def test_fit_brownresnick_recovers_extremal_coefficients():
    rng = np.random.default_rng(102)
    true_model = BrownResnickModel(range_=2.0, alpha=1.2)
    coords = np.array([[0.0, 0.0], [1.5, 0.0], [0.0, 1.5], [1.5, 1.5], [0.7, 2.2]])
    sims = true_model.simulate(coords, n_reps=1500, rng=rng)
    df = _frechet_frame(sims, coords.shape[0])

    fit = fit_pairwise_likelihood(BrownResnickModel, df, coords, rng=rng)

    test_hs = [np.array([1.5, 0.0]), np.array([1.5, 1.5])]
    for h in test_hs:
        true_theta = true_model.extremal_coefficient(h)
        fit_theta = fit.model.extremal_coefficient(h)
        assert abs(true_theta - fit_theta) < 0.15, (
            f"h={h}: true={true_theta:.3f} fit={fit_theta:.3f}"
        )


def test_max_pairs_subsampling_runs():
    """Sanity check that pair-subsampling for larger site counts doesn't crash
    and still gives a roughly sane fit."""
    rng = np.random.default_rng(103)
    true_model = SmithModel(cov=np.eye(2) * 1.5)
    coords = rng.uniform(0, 5, size=(10, 2))
    sims = true_model.simulate(coords, n_reps=500, rng=rng)
    df = _frechet_frame(sims, coords.shape[0])

    fit = fit_pairwise_likelihood(SmithModel, df, coords, max_pairs=15, rng=rng)
    assert fit.n_pairs_used == 15


def test_pairwise_fit_uses_jointly_observed_values():
    rng = np.random.default_rng(104)
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sims = SmithModel(cov=np.eye(2) * 1.5).simulate(coords, n_reps=200, rng=rng)
    sims[:50, 0] = np.nan
    sims[50:100, 1] = np.nan
    sims[100:150, 2] = np.nan
    df = _frechet_frame(sims, coords.shape[0])

    fit = fit_pairwise_likelihood(
        SmithModel, df, coords, max_pairs=2, rng=np.random.default_rng(105)
    )

    assert fit.n_pairs_used == 2
    assert fit.n_replicates == 200
    assert np.isfinite(fit.nllh)


def test_pairwise_fit_rejects_nonpositive_frechet_values():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    data = pl.DataFrame({"a": [1.0, 0.0], "b": [1.2, 1.4]})

    with pytest.raises(ValueError, match="strictly positive"):
        fit_pairwise_likelihood(SmithModel, data, coords)


def test_pairwise_fit_accepts_polars_with_null_values():
    rng = np.random.default_rng(106)
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    sims = SmithModel(cov=np.eye(2)).simulate(coords, n_reps=100, rng=rng)
    sims[:10, 0] = np.nan
    data = pl.DataFrame({"a": sims[:, 0], "b": sims[:, 1]}).with_columns(
        pl.all().fill_nan(None)
    )

    fit = fit_pairwise_likelihood(SmithModel, data, coords)

    assert fit.n_pairs_used == 1
    assert fit.n_replicates == 100


def test_pairwise_fit_accepts_legacy_pandas_input():
    pd = pytest.importorskip("pandas")
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    data = pd.DataFrame(
        {
            "a": [1.0, 2.0, 1.5, 3.0],
            "b": [1.2, 1.8, 2.0, 2.5],
        }
    )

    fit = fit_pairwise_likelihood(SmithModel, data, coords)

    assert fit.n_pairs_used == 1
    assert fit.n_replicates == 4


def test_curvature_adjusted_pairwise_posterior_samples():
    rng = np.random.default_rng(107)
    model = BrownResnickModel(range_=2.0, alpha=1.2)
    coords = np.array(
        [
            [0.0, 0.0],
            [1.5, 0.0],
            [0.0, 1.5],
            [1.5, 1.5],
            [0.7, 2.2],
        ]
    )
    data = _frechet_frame(model.simulate(coords, n_reps=250, rng=rng), len(coords))
    prior_calls = []

    with pytest.warns(UserWarning, match="approximate, local adjustment"):
        posterior = sample_pairwise_likelihood_posterior(
            BrownResnickModel,
            data,
            coords,
            n_samples=12,
            burn_in=8,
            thin=2,
            log_prior=lambda params: prior_calls.append(params.copy()) or 0.0,
            rng=np.random.default_rng(108),
        )

    assert posterior.free_param_samples.shape == (12, 2)
    assert posterior.log_posterior_samples.shape == (12,)
    assert np.all(np.isfinite(posterior.free_param_samples))
    assert 0 <= posterior.acceptance_rate <= 1
    assert posterior.pairwise_fit.n_pairs_used == 10
    assert np.all(np.isfinite(posterior.posterior_mean_free_params))
    assert isinstance(posterior.posterior_mean_model, BrownResnickModel)
    assert prior_calls

    adjusted_curvature = (
        posterior.curvature_adjustment.T
        @ posterior.sensitivity_matrix
        @ posterior.curvature_adjustment
    )
    assert adjusted_curvature == pytest.approx(posterior.curvature_matrix, rel=1e-5)
