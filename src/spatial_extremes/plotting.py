"""Optional plots for spatial fields and extreme-value dependence."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import polars as pl

from .dependence import _to_numpy, fmadogram


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires Matplotlib; install it with "
            '`pip install "spatial-extremes[plotting]"`.'
        ) from exc
    return plt


def plot_spatial_field(
    values,
    coords: np.ndarray,
    *,
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    marker_size: float = 60,
    colorbar_label: str = "Value",
):
    """Plot a scalar field at irregularly spaced 2-D sites.

    Parameters
    ----------
    values : array_like, shape (D,)
        One observed or simulated value per site.
    coords : array_like, shape (D, 2)
        Projected planar coordinates, in any consistent linear units.
    ax : matplotlib.axes.Axes, optional
        Existing axes. If omitted, a new figure and axes are created.
    title : str, optional
        Plot title.
    cmap : str
        Matplotlib colormap name.
    marker_size : float
        Scatter marker area in points squared.
    colorbar_label : str
        Label for the value colorbar.

    Returns
    -------
    (figure, axes)
        The Matplotlib figure and axes. The figure can be saved with
        ``figure.savefig(...)``; the function does not call ``show``.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_sites, 2)")
    if values.ndim != 1 or values.shape[0] != coords.shape[0]:
        raise ValueError("values must be one-dimensional with one value per site")
    if coords.shape[0] == 0 or not np.isfinite(coords).all():
        raise ValueError("coords must contain at least one site and only finite values")
    if np.isinf(values).any() or not np.isfinite(values).any():
        raise ValueError("values must contain at least one finite value and no infinities")
    if marker_size <= 0:
        raise ValueError("marker_size must be positive")

    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    points = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        cmap=cmap,
        s=marker_size,
    )
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label(colorbar_label)
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_aspect("equal", adjustable="datalim")
    if title is not None:
        ax.set_title(title)
    return fig, ax


def plot_spatial_field_map(
    values,
    lon_lat: np.ndarray,
    *,
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    marker_size: float = 60,
    colorbar_label: str = "Value",
    source=None,
    zoom: int | str = "auto",
):
    """Plot a scalar field at longitude/latitude sites over a web tile map.

    The locations are transformed from WGS 84 longitude/latitude (EPSG:4326)
    to Web Mercator (EPSG:3857) for display. Fetching the map tiles requires an
    internet connection. This visualization projection is for plotting only;
    use suitable projected coordinates for spatial model fitting and distance
    calculations.

    Parameters
    ----------
    values : array_like, shape (D,)
        One observed or simulated value per site.
    lon_lat : array_like, shape (D, 2)
        Geographic coordinates in ``[longitude, latitude]`` order, in degrees.
        Longitudes must be in [-180, 180] and latitudes in the Web Mercator
        display range [-85.0511, 85.0511].
    ax : matplotlib.axes.Axes, optional
        Existing axes. If omitted, a new figure and axes are created.
    title : str, optional
        Plot title.
    cmap : str
        Matplotlib colormap name.
    marker_size : float
        Scatter marker area in points squared.
    colorbar_label : str
        Label for the value colorbar.
    source : contextily tile provider or URL, optional
        Tile source. Defaults to OpenStreetMap's standard tile layer.
    zoom : int or "auto"
        Tile zoom level passed to Contextily.

    Returns
    -------
    (figure, axes)
        The Matplotlib figure and axes. The figure can be saved with
        ``figure.savefig(...)``; the function does not call ``show``.

    Notes
    -----
    Install the optional map dependencies with
    ``pip install "spatial-extremes[maps]"``. Tile availability, usage terms,
    and attribution requirements are set by the selected tile provider.
    """
    lon_lat = np.asarray(lon_lat, dtype=float)
    values = np.asarray(values, dtype=float)
    if lon_lat.ndim != 2 or lon_lat.shape[1] != 2:
        raise ValueError("lon_lat must have shape (n_sites, 2) in [longitude, latitude] order")
    if values.ndim != 1 or values.shape[0] != lon_lat.shape[0]:
        raise ValueError("values must be one-dimensional with one value per site")
    if lon_lat.shape[0] == 0 or not np.isfinite(lon_lat).all():
        raise ValueError("lon_lat must contain at least one site and only finite values")
    if np.any(np.abs(lon_lat[:, 0]) > 180):
        raise ValueError("longitude values must be in [-180, 180] degrees")
    if np.any(np.abs(lon_lat[:, 1]) > 85.0511):
        raise ValueError("latitude values must be in [-85.0511, 85.0511] for Web Mercator")
    if np.isinf(values).any() or not np.isfinite(values).any():
        raise ValueError("values must contain at least one finite value and no infinities")
    if marker_size <= 0:
        raise ValueError("marker_size must be positive")

    try:
        import contextily as ctx
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "Map plotting requires Contextily and GeoPandas; install them with "
            '`pip install "spatial-extremes[maps]"`.'
        ) from exc

    plt = _pyplot()
    points = gpd.GeoDataFrame(
        {"value": values},
        geometry=gpd.points_from_xy(lon_lat[:, 0], lon_lat[:, 1]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    artist = ax.scatter(
        points.geometry.x,
        points.geometry.y,
        c=values,
        cmap=cmap,
        s=marker_size,
        zorder=2,
    )
    colorbar = fig.colorbar(artist, ax=ax)
    colorbar.set_label(colorbar_label)
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik if source is None else source,
        crs=points.crs,
        zoom=zoom,
    )
    ax.set_xlabel("Web Mercator easting (m)")
    ax.set_ylabel("Web Mercator northing (m)")
    ax.set_aspect("equal", adjustable="datalim")
    if title is not None:
        ax.set_title(title)
    return fig, ax


def plot_extremal_dependence(
    frechet_df,
    coords: np.ndarray,
    *,
    model=None,
    statistic: str = "extremal_coefficient",
    n_bins: int = 10,
    ax=None,
):
    """Plot binned empirical extremal dependence, optionally against a model.

    Empirical values are calculated from complete-case unit-Frechet
    observations using the F-madogram. ``statistic`` can be
    ``"extremal_coefficient"`` or ``"fmadogram"``. When supplied, ``model``
    must provide ``extremal_coefficient(h)`` for a two-dimensional
    displacement vector; its theoretical values are averaged over site pairs
    in each distance bin, preserving anisotropy for models such as Smith.

    Parameters
    ----------
    frechet_df : Polars DataFrame, or (with the ``pandas`` extra installed)
        pandas DataFrame. Rows are independent replicates and columns are
        sites. Values must be positive unit-Frechet observations; incomplete
        rows are dropped.
    coords : array_like, shape (n_sites, 2)
        Projected coordinates in the same column order as ``frechet_df``.
    model : fitted model, optional
        Model instance exposing ``extremal_coefficient(displacement)``.
    statistic : {"extremal_coefficient", "fmadogram"}
        Dependence summary to plot against distance.
    n_bins : int
        Number of equal-width distance bins.
    ax : matplotlib.axes.Axes, optional
        Existing axes. If omitted, a new figure and axes are created.

    Returns
    -------
    (figure, axes)
        The Matplotlib figure and axes. The function does not call ``show``.
    """
    if statistic not in {"extremal_coefficient", "fmadogram"}:
        raise ValueError("statistic must be 'extremal_coefficient' or 'fmadogram'")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    values, columns = _to_numpy(frechet_df)
    values = np.asarray(values, dtype=float)
    coords = np.asarray(coords, dtype=float)
    if values.ndim != 2:
        raise ValueError("frechet_df must be a two-dimensional table")
    if coords.shape != (values.shape[1], 2) or not np.isfinite(coords).all():
        raise ValueError(
            f"coords must be a finite array with shape ({values.shape[1]}, 2)"
        )
    if values.shape[1] < 2:
        raise ValueError("At least two sites are required to plot spatial dependence")
    observed = ~np.isnan(values)
    if np.isinf(values).any() or np.any(values[observed] <= 0):
        raise ValueError("Observed unit-Frechet values must be finite and positive")
    complete = values[np.isfinite(values).all(axis=1)]
    if complete.shape[0] == 0:
        raise ValueError("frechet_df has no complete finite rows")

    complete_df = pl.DataFrame(complete, schema=columns, orient="row")
    pairs = fmadogram(complete_df, coords)
    pair_indices = list(combinations(range(len(coords)), 2))
    distances = np.array([
        np.linalg.norm(coords[i] - coords[j])
        for i, j in pair_indices
    ])
    bin_edges = np.linspace(distances.min(), distances.max(), int(n_bins) + 1)
    bin_ids = np.digitize(distances, bin_edges[1:-1], right=False)
    pairs = pairs.with_columns(
        pl.Series("distance", distances),
        pl.Series("dist_bin", bin_ids),
    )
    binned = (
        pairs.lazy()
        .group_by("dist_bin")
        .agg(
            pl.col("distance").mean().alias("mean_distance"),
            pl.col("fmadogram").mean().alias("mean_fmadogram"),
            pl.col("extremal_coefficient").mean().alias("mean_extremal_coefficient"),
        )
        .sort("dist_bin")
        .collect()
    )
    y_column = (
        "mean_extremal_coefficient"
        if statistic == "extremal_coefficient"
        else "mean_fmadogram"
    )

    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(
        binned.get_column("mean_distance").to_numpy(),
        binned.get_column(y_column).to_numpy(),
        marker="o",
        linestyle="none",
        label="Empirical",
    )
    if model is not None:
        pairs = pairs.with_columns(
            pl.Series(
                "model_extremal_coefficient",
                [
                    model.extremal_coefficient(coords[i] - coords[j])
                    for i, j in pair_indices
                ],
            )
        )
        if statistic == "fmadogram":
            theta = pairs.get_column("model_extremal_coefficient")
            pairs = pairs.with_columns(
                ((theta - 1) / (2 * (theta + 1))).alias("model_fmadogram")
            )
            model_column = "model_fmadogram"
        else:
            model_column = "model_extremal_coefficient"
        theoretical = (
            pairs.lazy()
            .group_by("dist_bin")
            .agg(
                pl.col("distance").mean().alias("mean_distance"),
                pl.col(model_column).mean().alias("model_value"),
            )
            .sort("dist_bin")
            .collect()
        )
        ax.plot(
            theoretical.get_column("mean_distance").to_numpy(),
            theoretical.get_column("model_value").to_numpy(),
            label=type(model).__name__,
        )

    ax.set_xlabel("Distance")
    ax.set_ylabel(
        "Extremal coefficient" if statistic == "extremal_coefficient" else "F-madogram"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax
