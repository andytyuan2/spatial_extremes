"""
Model-free spatial dependence diagnostics for max-stable data.

These work directly on unit-Frechet-transformed observations and let
you sanity-check a fitted model (e.g. Smith) against the empirical
dependence structure before trusting parameter estimates.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def _to_numpy(df) -> tuple[np.ndarray, list[str]]:
    """Return values and column names from a Polars or optional pandas table."""
    if isinstance(df, pl.DataFrame):
        return df.to_numpy(), df.columns
    if type(df).__module__.split(".", maxsplit=1)[0] == "pandas":
        return df.to_numpy(), [str(column) for column in df.columns]
    raise TypeError("input must be a Polars DataFrame or a pandas DataFrame")


def fmadogram(frechet_df, coords: np.ndarray) -> pl.DataFrame:
    """F-madogram: pairwise dependence summary on unit-Frechet margins.

        nu_F(h) = 0.5 * E[ |F(Z(s)) - F(Z(s+h))| ]

    where F is the unit-Frechet CDF, so F(Z(s)) is uniform on [0,1]
    marginally. Related to the extremal coefficient by:

        theta(h) = (1 + 2*nu_F(h)) / (1 - 2*nu_F(h))

    Parameters
    ----------
    frechet_df : Polars DataFrame, or (with the ``pandas`` extra installed)
        pandas DataFrame. Columns are sites, rows are replicates, already on
        unit-Frechet margins (see gev.GEVResult.to_unit_frechet).
    coords : (D, 2) array_like of site coordinates, same order as columns.

    Returns
    -------
    Polars DataFrame with one row per site pair: site_i, site_j, distance,
    fmadogram, extremal_coefficient.
    """
    values, cols = _to_numpy(frechet_df)
    coords = np.asarray(coords, dtype=float)
    _, D = values.shape
    if coords.shape[0] != D:
        raise ValueError(f"coords has {coords.shape[0]} rows but frechet_df has {D} columns")

    # Uniform-margin transform via unit-Frechet CDF: F(z) = exp(-1/z)
    u = np.exp(-1.0 / values)

    rows = []
    for i in range(D):
        for j in range(i + 1, D):
            nu = 0.5 * np.mean(np.abs(u[:, i] - u[:, j]))
            theta = (1 + 2 * nu) / (1 - 2 * nu) if nu < 0.5 else 2.0
            dist = np.linalg.norm(coords[i] - coords[j])
            rows.append({
                "site_i": cols[i], "site_j": cols[j],
                "distance": dist, "fmadogram": nu,
                "extremal_coefficient": theta,
            })
    return pl.DataFrame(rows)


def empirical_extremal_coefficient(frechet_df, coords: np.ndarray) -> pl.DataFrame:
    """Empirical extremal coefficient via the madogram-based estimator.

    Thin wrapper around fmadogram() returning just the columns most
    people plot (distance vs extremal_coefficient), useful for
    overlaying against a fitted model's `extremal_coefficient(h)`.
    """
    result = fmadogram(frechet_df, coords)
    return (
        result.lazy()
        .select("site_i", "site_j", "distance", "extremal_coefficient")
        .collect()
    )


def binned_extremal_coefficient(frechet_df, coords: np.ndarray, n_bins: int = 10) -> pl.DataFrame:
    """Bin pairwise extremal coefficients by distance -- handy for plotting
    a smooth empirical curve against a fitted theoretical one."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    ec = empirical_extremal_coefficient(frechet_df, coords)
    distances = ec.get_column("distance").to_numpy()
    bin_edges = np.linspace(distances.min(), distances.max(), int(n_bins) + 1)
    bin_ids = np.digitize(distances, bin_edges[1:-1], right=False)
    summary = (
        ec.with_columns(pl.Series("dist_bin", bin_ids))
        .lazy()
        .group_by("dist_bin")
        .agg(
            pl.col("distance").mean().alias("mean_distance"),
            pl.col("extremal_coefficient").mean().alias("mean_extremal_coefficient"),
            pl.len().alias("n_pairs"),
        )
        .sort("dist_bin")
        .drop("dist_bin")
        .collect()
    )
    return summary
