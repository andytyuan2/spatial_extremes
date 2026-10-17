# spatial-extremes

`spatial-extremes` is a Python package for working with spatial extremes. It
provides tools to fit generalized extreme-value (GEV) distributions to block
maxima, model spatial dependence with max-stable processes, simulate spatial
fields, and summarize extremal dependence.

The package is intended for researchers and practitioners analyzing
replicated extremes observed at multiple spatial locations. It implements
extreme-value models and methods in Python; it is not a port of R's
`SpatialExtremes`.

## Features

- Fit stationary GEV distributions to individual series or sites.
- Transform observations to unit-Frechet margins for spatial analysis.
- Simulate Smith, Schlather, and Brown-Resnick max-stable models.
- Fit spatial dependence parameters using pairwise composite likelihood.
- Calculate empirical F-madograms and extremal-coefficient summaries.
- Optionally plot spatial fields and distance-binned dependence.
- Optionally use an amortized neural estimator for isotropic
  Brown-Resnick and Schlather models.

## Installation

Requires Python 3.10 or later. Install the core package with:

```bash
python -m pip install spatial-extremes
```

Optional dependencies are available as extras:

| Extra | Adds |
| --- | --- |
| `geo` | GeoPandas support for `SmithModel.simulate_geo` |
| `pandas` | pandas DataFrame input for tabular routines |
| `neural` | Neural estimator dependencies (joblib and scikit-learn) |
| `plotting` | Matplotlib plots |
| `maps` | Geographic plots over online map tiles |

Install an extra, or combine several extras, like this:

```bash
python -m pip install "spatial-extremes[plotting]"
python -m pip install "spatial-extremes[geo,plotting,maps]"
```

The core package uses NumPy, SciPy, and Polars. Tabular routines accept and
return Polars data; NumPy arrays are also supported by relevant functions.
The `maps` plotting helper retrieves web map tiles and therefore requires an
internet connection.

## Quick start: simulate a spatial field

Model coordinates must be two-dimensional Euclidean coordinates, expressed
in consistent linear units. The simulator returns unit-Frechet observations
with shape `(number of replicates, number of sites)`.

```python
import numpy as np
from spatial_extremes.models import BrownResnickModel

coordinates = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

model = BrownResnickModel(range_=1.0, alpha=1.0)
fields = model.simulate(
    coordinates,
    n_reps=100,
    rng=np.random.default_rng(42),
)

print(fields.shape)  # (100, 4)
```

## Typical analysis workflow

For observed block maxima, a common workflow is:

1. Store each independent replicate (for example, each year) as a row and
   each site as a column.
2. Fit a GEV distribution to each site's block maxima using `fit_gev` or
   `fit_gev_by_site`.
3. Transform each site's observations to unit-Frechet margins with
   `GEVResult.to_unit_frechet`.
4. Fit a spatial dependence model with `fit_pairwise_likelihood`, or
   inspect empirical dependence with `fmadogram`,
   `empirical_extremal_coefficient`, or `binned_extremal_coefficient`.

When fitting spatial dependence, the coordinate rows must match the site
columns in order. Use projected coordinates rather than longitude and
latitude degrees. See the [How-to Guide](https://github.com/andytyuan2/spatial_extremes/blob/main/HOW_TO.md) for a complete
data-preparation and fitting example.

## Available models and methods

The `spatial_extremes` package exports `SmithModel`, `SchlatherModel`, and
`BrownResnickModel`. Each model provides bivariate distribution and
dependence methods, as well as spatial simulation. Pairwise fitting is
available through `fit_pairwise_likelihood`.

Other top-level functions include:

- `fit_gev` and `fit_gev_by_site` for stationary marginal GEV fits.
- `fmadogram`, `empirical_extremal_coefficient`, and
  `binned_extremal_coefficient` for empirical dependence summaries.

Optional plotting functions are in `spatial_extremes.plotting`. The optional
neural estimator is imported separately from
`spatial_extremes.neural_estimator`.

## Assumptions and limitations

- GEV fits assume independent, identically distributed observations from a
  stationary distribution at each site. They do not model trends, covariates,
  or spatial pooling of marginal parameters.
- Pairwise likelihood assumes independent replicates from a stationary
  spatial process. It is a composite likelihood; uncertainty from the
  marginal fits is not propagated, and composite-likelihood standard errors
  are not provided.
- Missing values are supported by the pairwise fit, which uses replicates
  observed at both sites in each pair. The missingness mechanism must be
  ignorable for the dependence being estimated. Empirical madogram and
  extremal-coefficient summaries require complete, positive unit-Frechet
  observations.
- Coordinates are not reprojected by the modeling or fitting routines.
  Distances and model ranges use the units of the supplied coordinates.
- Simulation approximates an infinite max-stable construction using a
  finite storm-center window or a probabilistic stopping bound, depending
  on the model. A maximum-points cap can also stop simulation early, so
  simulated values can have a small truncation error.
- Model fit and successful optimizer convergence do not establish that the
  assumptions or the selected model are appropriate for a dataset.

See [Technical Details](https://github.com/andytyuan2/spatial_extremes/blob/main/TECHNICAL_DETAILS.md) for model equations,
statistical assumptions, and implementation details.

## Example data

The repository includes example data and a generator in
[`Dummy_data/`](https://github.com/andytyuan2/spatial_extremes/tree/main/Dummy_data).
The generated observations use a Gaussian copula with covariate-dependent
GEV margins, a time trend, and randomly missing values. They are not
generated from a max-stable process, so they are suitable for exploring
workflows and model misspecification, not for validating recovery of known
max-stable parameters. The generator and data files are available in the
repository, not as installed package resources.

## References

The implementation draws on the extreme-value methods of Smith (1990),
Schlather (2002), Kabluchko, Schlather & de Haan (2009), and Padoan, Ribatet
& Sisson (2010). The package is independently implemented and is not copied
from the GPL-licensed R `SpatialExtremes` source.

## Documentation and support

- [How-to Guide](https://github.com/andytyuan2/spatial_extremes/blob/main/HOW_TO.md): end-to-end examples and practical guidance.
- [Technical Details](https://github.com/andytyuan2/spatial_extremes/blob/main/TECHNICAL_DETAILS.md): formulas and assumptions.
- [Releases](https://github.com/andytyuan2/spatial_extremes/releases)
- [Report an issue](https://github.com/andytyuan2/spatial_extremes/issues)
