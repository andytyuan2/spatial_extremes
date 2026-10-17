import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("matplotlib.pyplot")

from spatial_extremes.models import BrownResnickModel
from spatial_extremes.plotting import (
    plot_extremal_dependence,
    plot_spatial_field,
    plot_spatial_field_map,
)


def test_plot_spatial_field_returns_figure_and_colored_sites():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    values = np.array([1.0, 2.0, 3.0])

    fig, ax = plot_spatial_field(values, coords, title="Example field")

    assert ax.get_title() == "Example field"
    assert ax.collections[0].get_offsets().shape == (3, 2)
    assert len(fig.axes) == 2  # plot axes and colorbar axes
    matplotlib.pyplot.close(fig)


def test_plot_extremal_dependence_includes_empirical_and_model_curves():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    values = pd.DataFrame([
        [1.0, 2.0, 3.0],
        [2.0, 2.5, 1.5],
        [4.0, 3.0, 5.0],
        [3.0, 4.0, 2.0],
    ])
    model = BrownResnickModel(range_=2.0, alpha=1.0)

    fig, ax = plot_extremal_dependence(
        values,
        coords,
        model=model,
        statistic="extremal_coefficient",
        n_bins=2,
    )

    assert ax.get_ylabel() == "Extremal coefficient"
    assert len(ax.lines) == 2
    assert [line.get_label() for line in ax.lines] == [
        "Empirical",
        "BrownResnickModel",
    ]
    matplotlib.pyplot.close(fig)


def test_plot_extremal_dependence_drops_incomplete_rows():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    values = pd.DataFrame([
        [1.0, 2.0, 3.0],
        [2.0, np.nan, 1.5],
        [4.0, 3.0, 5.0],
    ])

    fig, ax = plot_extremal_dependence(
        values,
        coords,
        statistic="fmadogram",
        n_bins=2,
    )

    assert ax.get_ylabel() == "F-madogram"
    assert len(ax.lines) == 1
    matplotlib.pyplot.close(fig)


def test_plot_spatial_field_map_projects_lon_lat_and_adds_tiles(monkeypatch):
    contextily = pytest.importorskip("contextily")
    pytest.importorskip("geopandas")

    basemap_calls = []

    def fake_add_basemap(ax, **kwargs):
        basemap_calls.append((ax, kwargs))

    monkeypatch.setattr(contextily, "add_basemap", fake_add_basemap)
    fig, ax = plot_spatial_field_map(
        [1.0, 2.0],
        [[0.0, 0.0], [1.0, 1.0]],
        source="https://tiles.example/{z}/{x}/{y}.png",
        zoom=5,
    )

    offsets = ax.collections[0].get_offsets()
    np.testing.assert_allclose(offsets[0], [0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(offsets[1], [111319.49, 111325.14], atol=1.0)
    assert basemap_calls[0][0] is ax
    assert basemap_calls[0][1]["source"] == "https://tiles.example/{z}/{x}/{y}.png"
    assert basemap_calls[0][1]["zoom"] == 5
    assert basemap_calls[0][1]["crs"].to_epsg() == 3857
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize(
    ("values", "coords", "message"),
    [
        ([1.0, 2.0], [[0.0, 0.0]], "one value per site"),
        ([1.0], [[0.0, 0.0, 0.0]], "shape"),
        ([np.inf], [[0.0, 0.0]], "finite value"),
    ],
)
def test_plot_spatial_field_validates_inputs(values, coords, message):
    with pytest.raises(ValueError, match=message):
        plot_spatial_field(values, coords)


def test_plot_extremal_dependence_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="positive"):
        plot_extremal_dependence(
            pd.DataFrame([[1.0, 0.0], [2.0, 3.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
        )


@pytest.mark.parametrize(
    ("coords", "message"),
    [
        ([[181.0, 0.0]], "longitude"),
        ([[0.0, 86.0]], "latitude"),
    ],
)
def test_plot_spatial_field_map_validates_geographic_coordinates(coords, message):
    with pytest.raises(ValueError, match=message):
        plot_spatial_field_map([1.0], coords)
