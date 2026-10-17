# spatial-extremes

Python tools for simulating and fitting spatial max-stable models and
for analyzing block maxima. The implementation follows published
extreme-value formulas; it is not a port of R's `SpatialExtremes`.

## Installation

Install the core package and its required dependencies:

```bash
pip install spatial-extremes
```

Optional dependencies are grouped by feature:

```bash
pip install "spatial-extremes[geo]"      # GeoDataFrame helper
pip install "spatial-extremes[pandas]"   # legacy pandas DataFrame compatibility
pip install "spatial-extremes[neural]"   # amortized neural estimator
pip install "spatial-extremes[plotting]" # Matplotlib plots
pip install "spatial-extremes[maps]"     # Geographic plots over web map tiles
pip install "spatial-extremes[dev]"      # tests, linting, and builds
```

The core dependencies are NumPy, SciPy, and Polars. `pandas` enables legacy
pandas DataFrame input for tabular routines; core tabular operations and
results use Polars. `geo` installs GeoPandas for `SmithModel.simulate_geo`;
and `neural` installs joblib and scikit-learn for
the optional estimator in `spatial_extremes.neural_estimator`. Polars handles
CSV and Parquet I/O without a separate data extra. `plotting` installs Matplotlib for optional field and
dependence plots; `maps` installs GeoPandas and Contextily for geographic
field plots over web map tiles. These extras are not needed by the core
fitting and simulation APIs.

For a local development install with every optional feature:

```bash
pip install -e ".[geo,pandas,neural,plotting,maps,dev]"
```

## Package features

- `spatial_extremes.gev`: univariate GEV maximum-likelihood fitting
  (`fit_gev`, `fit_gev_by_site`) and probability-integral transformation
  to unit-Frechet margins. Table results use Polars.
- `spatial_extremes.models`: Smith, Schlather, and Brown-Resnick model
  simulation, bivariate CDFs and log-densities, and theoretical
  extremal coefficients.
- `spatial_extremes.dependence`: empirical F-madogram and
  madogram-based extremal-coefficient summaries.
- `spatial_extremes.plotting`: optional Matplotlib plots of spatial fields
  and distance-binned extremal dependence; its map helper additionally
  requires the `maps` extra and an internet connection for tiles.
- `spatial_extremes.fitting`: pairwise composite-likelihood estimation
  of spatial dependence parameters.
- `spatial_extremes.neural_estimator`: optional amortized estimator for
  isotropic Brown-Resnick and Schlather models. It is imported separately
  so the core package does not require scikit-learn or joblib.

## Typical workflow

For block maxima, fit a GEV at each site, transform the observations to
unit-Frechet margins, then fit or inspect the spatial dependence. Marginal
fits can be made from wide tables (one column per site) or long tables:

```python
from spatial_extremes import fit_gev_by_site

gev_fits = fit_gev_by_site(
    observations,
    site_col="station_id",
    value_col="annual_max_mm",
)
```

For pairwise spatial fitting, pass a wide table of unit-Frechet values:
columns must match the order of the coordinate rows, and coordinates
must have shape `(number_of_sites, 2)`. Each site pair uses only
replicates observed at both sites. Missing values are therefore
permitted, but the likelihood is valid only when the missingness
mechanism is ignorable for the dependence being estimated. Infinite
values and non-positive observed unit-Frechet values are invalid.

## Statistical assumptions and limitations

### Margins and replicates

- `fit_gev` treats the observations at one site as independent,
  identically distributed block maxima from a stationary GEV
  distribution. It requires at least 10 non-missing observations; this
  is a practical guard, not a guarantee of reliable tail estimates.
  Reported standard errors are approximations from a numerically
  differentiated likelihood Hessian.
- `fit_gev_by_site` fits a separate stationary GEV at each site. It does
  not model time trends, covariates, or spatial pooling of marginal
  parameters.
- `GEVResult.to_unit_frechet` uses the fitted GEV CDF and clips
  probabilities to avoid infinities. The resulting margins are
  estimated, not known, unit-Frechet margins.
- The pairwise likelihood treats rows as independent replicates of one
  stationary spatial process. Its objective is a composite likelihood,
  not the full joint likelihood. It reports the optimizer result but
  does not provide composite-likelihood standard errors or account for
  uncertainty from the preceding marginal fits.
- `max_pairs` randomly selects a fixed subset of eligible site pairs
  before optimization. This reduces computation but makes the result
  dependent on the chosen subset and random seed.

### Spatial coordinates and model validity

- Model coordinates are two-dimensional Euclidean coordinates in
  consistent linear units. The routines do not reproject coordinates.
  Do not use longitude/latitude degrees as if they were distances;
  project the locations first. The reported distance and model range
  use the same coordinate units.
- Smith covariance matrices must be positive definite. Simulation uses
  a finite storm-center window expanded by `buffer_factor`; the finite
  window approximates contributions from storms outside the window.
  `max_points` is a hard cap and can stop simulation before the normal
  stopping condition is reached.
- Schlather's correlation function must describe a valid stationary
  Gaussian correlation model. The default is powered exponential; a
  custom `correlation_fn` is used by both the bivariate formulas and the
  simulator. The resulting site correlation matrix must be positive
  semidefinite or simulation's Cholesky factorization will fail. Its
  extremal coefficient cannot exceed `1 + sqrt(2)/2` (about 1.707), so
  this model cannot attain independence at large separation.
- Brown-Resnick's default power variogram is
  `gamma(h) = (|h| / range_)**alpha`, with positive `range_` and
  `0 < alpha <= 2`. A custom variogram must be conditionally
  negative-definite and zero at zero lag to define a valid model; the
  package does not validate these properties for a user-supplied
  function.
- Schlather and Brown-Resnick simulation truncate an infinite Poisson
  construction using a probabilistic stopping bound; `max_points` is
  also a hard cap. Thus, unlike an exact infinite construction, results
  can have a small truncation error.

The F-madogram and extremal-coefficient diagnostics expect positive,
complete unit-Frechet observations and do not filter missing values.
They are descriptive summaries, not fitted-model goodness-of-fit tests.

### Optional neural estimator

The neural estimator targets isotropic Brown-Resnick or Schlather models
and must be trained for the same station layout and replicate count used
at prediction time. It learns under a built-in parameter prior, so its
predictions are prior-dependent point estimates. `predict_bootstrap`
resamples replicate rows to describe resampling variability; it is not a
posterior distribution or a substitute for calibrated uncertainty
estimation.

## Supplied example data

`Dummy_data/gen_extremes.py` creates long-format station/year
observations and station-level truth metadata. Its observations come
from a Gaussian copula with an exponential spatial correlation,
station-specific covariate-dependent GEV parameters, a time trend in GEV
location, and randomly missing values. This is **not** a max-stable
process and its margins are not stationary over time. Fitting the
package's stationary max-stable models to this data is a workflow or
misspecification exercise, not a test of recovery of a true max-stable
range or variogram.

The generator and example files are in the repository checkout, not
included as installed package resources. Polars is a core dependency, so run
`python gen_extremes.py` from the `Dummy_data` directory to regenerate the
Parquet files. The output files are written to the current working directory.

## Validation and scope

See [Technical Details](./TECHNICAL_DETAILS.md) for the model equations
and the precise assumptions behind fitting and simulation.
For a step-by-step introduction and a more advanced workflow, see the
[How-to Guide](./HOW_TO.md).

The test suite checks selected model properties, including simulated
margins, Monte Carlo agreement with bivariate CDFs, and recovery of
dependence summaries on simulated data. These tests exercise specific
configurations; they are not a proof of correctness for every parameter,
coordinate layout, or custom function.

Marginal GEV parameters are currently fitted separately from spatial
dependence (a two-step workflow). Joint marginal/dependence fitting,
conditional simulation, and non-stationary variogram parameterizations
are not implemented.

## References

The model formulas are based on Smith (1990), Schlather (2002),
Kabluchko, Schlather & de Haan (2009), and Padoan, Ribatet & Sisson
(2010). The implementation is independently written rather than copied
from the GPL-licensed R `SpatialExtremes` source.
