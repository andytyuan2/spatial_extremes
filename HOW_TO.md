# How to Use `spatial-extremes`

This guide has two paths through the package:

1. **A simple guide** that explains the workflow with minimal statistical
   background.
2. **A detailed guide** for users who want to understand the assumptions,
   likelihood, model choices, diagnostics, and optional neural estimator.

The examples assume you have block maxima observed at several spatial sites.
For example, each row might be one year's maximum daily rainfall and each
column one weather station.

## 1. Simple guide

### What the package does

Measurements at a site can have a different scale from measurements at another
site. The package handles this in two stages:

1. Fit a **GEV distribution** to the block maxima at each site and convert the
   observations to a common unit-Frechet scale.
2. Fit a **spatial max-stable model** to describe how often extremes occur
   together at different sites.

The GEV step describes the size of extremes at each individual site. The
spatial step describes dependence between sites. Neither step automatically
proves that its assumptions are right; use plots, domain knowledge, and
out-of-sample checks as well.

### Install

Install the core package:

```bash
python -m pip install spatial-extremes
```

For local development from a repository checkout, install the optional
dependencies you need. For example, this installs the neural estimator and
development tools too:

```bash
python -m pip install -e ".[neural,dev]"
```

Optional groups are `geo`, `pandas`, `neural`, `plotting`, `maps`, and
`dev`; see the installation section of the [README](./README.md). Install
`plotting` if you want to create Matplotlib figures:

```bash
python -m pip install "spatial-extremes[plotting]"
```

For geographic field plots over an online tile map, install the `maps` extra:

```bash
python -m pip install "spatial-extremes[maps]"
```

### Prepare the data

The regular workflow uses a **wide** table:

- one row per independent block or replicate (for example, one year);
- one numeric column per site;
- a separate coordinate array with one row per site, in the same order as
  those columns.

For this example, assume `annual_maxima` is a Polars DataFrame and
`coordinates` is a NumPy array with columns `x` and `y`. Coordinates should be
projected into a planar coordinate system, not longitude and latitude degrees.

```python
import numpy as np
import polars as pl

# Scan lazily and collect once before fitting.
annual_maxima = pl.scan_csv("annual_maxima.csv").drop("year").collect()

# The coordinate rows MUST be in the same order as annual_maxima.columns.
coordinates = (
    pl.scan_csv("station_coordinates.csv")
    .select("x", "y")
    .collect()
    .to_numpy()
)
```

Make the ordering explicit and check it before fitting:

```python
site_names = list(annual_maxima.columns)
assert coordinates.shape == (len(site_names), 2)
```

### Fit site-by-site GEV distributions and transform the observations

```python
from spatial_extremes import fit_gev

gev_fits = {
    site: fit_gev(annual_maxima.get_column(site))
    for site in site_names
}

# Convert each site's observations to the unit-Frechet scale.
# Missing values remain missing and are allowed in the pairwise fit below.
frechet = pl.DataFrame({
    site: gev_fits[site].to_unit_frechet(
        annual_maxima.get_column(site).to_numpy()
    )
    for site in site_names
})
```

`fit_gev` drops missing values, rejects infinite values, and requires at least
10 finite observations. Inspect the marginal estimates rather than assuming
they are reliable just because the optimizer returned:

```python
for site, result in gev_fits.items():
    print(site, result.loc, result.scale, result.shape, result.converged)
```

Here `loc`, `scale`, and `shape` are the GEV parameters. A failed convergence
flag or an implausible estimate is a reason to investigate the data and fit.
The package does not fit time trends or covariates in the GEV margins.

### Fit one spatial dependence model

This example fits a Brown-Resnick model using pairwise composite likelihood:

```python
from scipy.spatial.distance import pdist
from spatial_extremes import fit_pairwise_likelihood
from spatial_extremes.models import BrownResnickModel

# Use a starting range near the typical inter-site distance (in coordinate units).
initial_range = float(np.median(pdist(coordinates)))
fit = fit_pairwise_likelihood(
    BrownResnickModel,
    frechet,
    coordinates,
    x0_free=np.array([np.log(initial_range), 0.0]),
)

print("Converged:", fit.converged)
print("Range:", fit.model.range_)
print("Variogram exponent:", fit.model.alpha)
print("Site pairs used:", fit.n_pairs_used)
```

If convergence is false, do not treat the returned parameters as a successful
fit. Recheck data ordering and model suitability, and consider whether a
different optimizer starting point or a justified pair subsample is needed.

### Look at empirical dependence

The F-madogram provides a model-free summary of pairwise dependence. It expects
complete, positive unit-Frechet data, so this example retains only rows with
observations at every site:

```python
from spatial_extremes import binned_extremal_coefficient

complete_frechet = frechet.dropna()
empirical = binned_extremal_coefficient(
    complete_frechet,
    coordinates,
    n_bins=10,
)
print(empirical)
```

The table summarizes empirical extremal coefficients by distance. Compare
the broad pattern with the fitted model's theoretical dependence and with
scientific expectations. It is a diagnostic summary, not a formal model
adequacy test.
For extreme-value data this plots an F-madogram or extremal-coefficient
dependence curve, rather than the classical semivariogram often used for
Gaussian spatial processes.

Plot empirical extremal dependence and optionally overlay the fitted model:

```python
from spatial_extremes.plotting import plot_extremal_dependence

fig, ax = plot_extremal_dependence(
    frechet,
    coordinates,
    model=fit.model,
    statistic="extremal_coefficient",  # or "fmadogram"
    n_bins=10,
)
fig.savefig("extremal_dependence.png", dpi=200, bbox_inches="tight")
```

Incomplete rows are dropped before the empirical F-madogram calculation.
The model curve averages theoretical values over site pairs in each distance
bin, preserving directional differences for anisotropic models such as Smith.
The figure is returned and is not automatically displayed.

### Simulate a field

The model simulators return unit-Frechet values. For example:

```python
from spatial_extremes.models import BrownResnickModel

model = BrownResnickModel(range_=100_000, alpha=1.0)
simulated = model.simulate(
    coordinates,
    n_reps=20,
    rng=np.random.default_rng(42),
)
print(simulated.shape)  # (20, number of sites)
```

The range uses the same distance units as `coordinates`. A range of
`100_000` therefore means 100,000 units in the coordinate reference system
used for the sites.

Plot one simulated replicate at its site locations:

```python
from spatial_extremes.plotting import plot_spatial_field

fig, ax = plot_spatial_field(
    simulated[0],
    coordinates,
    title="Simulated unit-Frechet field",
    colorbar_label="Unit-Frechet value",
)
fig.savefig("spatial_field.png", dpi=200, bbox_inches="tight")
```

This is a colored scatter plot at the observed sites, not a spatial
interpolation to a continuous raster. Both plotting helpers require
Matplotlib, installed through the `plotting` extra.

### Overlay a spatial field on a map

If you have geographic coordinates, use `plot_spatial_field_map`. Supply
coordinates in **longitude, latitude** order, in degrees (WGS 84 / EPSG:4326).
The map helper reprojects them to Web Mercator for display and retrieves
OpenStreetMap tiles, so an internet connection is required:

```python
from spatial_extremes.plotting import plot_spatial_field_map

lon_lat = [
    [-123.12, 49.28],  # longitude, latitude for site 1
    [-122.33, 47.61],  # longitude, latitude for site 2
    [-121.89, 37.33],  # longitude, latitude for site 3
]
site_values = [2.1, 3.7, 1.4]

fig, ax = plot_spatial_field_map(
    site_values,
    lon_lat,
    title="Observed spatial field",
    colorbar_label="Field value",
)
fig.savefig("field_on_map.png", dpi=200, bbox_inches="tight")
```

The coordinates and values must have the same site ordering. This shows
colored markers at the measurement locations; it does not interpolate
between them. Tile providers control availability, terms of use, and any
attribution. For fitting spatial models or calculating distances, continue
to use coordinates projected into an appropriate local metric CRS; the
Web-Mercator conversion here is only for display. The display latitude must
be within approximately -85.05 to 85.05 degrees.

### When to use the neural estimator

The optional neural estimator is a separate, prior-dependent alternative for
**isotropic Brown-Resnick or Schlather models**. It is not automatically used
by `fit_pairwise_likelihood` or by the core package:

```python
from spatial_extremes.neural_estimator import NeuralEstimator

# This estimator's feature calculation does not accept missing observations.
# Its replicate count must match the complete rows supplied at prediction.
neural_frechet = frechet.dropna()
estimator = NeuralEstimator(
    "brownresnick",
    coords=coordinates,
    n_reps=len(neural_frechet),
    seed=42,
).fit(n_train=2000, n_jobs=1)

estimate = estimator.predict(neural_frechet.to_numpy())
print("Estimated range and alpha:", estimate)
```

Training simulates many datasets and can take time. This estimator is
simulation-based and Bayesian in the sense that it learns a prior-dependent
posterior-mean estimate; it does **not** return full posterior samples.
`predict_bootstrap` resamples replicate rows and describes resampling
variability, not posterior uncertainty. See [the detailed discussion below](#26-optional-neural-bayes-estimator).

## 2. Detailed guide

### 2.1 Data structure and spatial coordinates

Let $Y_{t,j}$ denote the block maximum for replicate $t$ at site $j$. The
package's standard table convention is:

$$
\mathrm{rows} = \mathrm{independent\ replicates}, \qquad
\mathrm{columns} = \mathrm{sites}.
$$

The coordinate array must have shape $(D,2)$ for $D$ sites. Coordinate row
$j$ must correspond to data column $j$. The package uses ordinary Euclidean
distances; use projected coordinates expressed in consistent linear units.
It does not reproject geographic coordinates.

The replicates are assumed to be independent and identically distributed
across rows for the stationary spatial likelihood. If there is temporal
dependence, a trend, changing measurement practice, or a changing climate
regime, address it before interpreting a stationary fit. The package does not
model those features automatically.

`fit_gev_by_site` also accepts long-format Polars data for marginal
fits. For example:

```python
from spatial_extremes import fit_gev_by_site

marginal_table = fit_gev_by_site(
    long_observations,
    site_col="station_id",
    value_col="annual_max",
)
```

This fits a separate GEV at each site. It does not itself transform the long
table or fit spatial dependence. For the dependence step, arrange transformed
values as a wide table with one column per site.

### 2.2 Marginal GEV model

At each site $j$, the stationary GEV CDF is

$$
G_j(y)
=
\exp\left[
-\left(
1+\xi_j\frac{y-\mu_j}{\sigma_j}
\right)^{-1/\xi_j}
\right],
\qquad
1+\xi_j\frac{y-\mu_j}{\sigma_j}>0,
$$

where $\mu_j$ is location, $\sigma_j>0$ is scale, and $\xi_j$ is shape.
The $\xi_j=0$ case is interpreted by the Gumbel limit. `fit_gev` estimates
these parameters by univariate maximum likelihood. SciPy uses the opposite
shape sign convention internally; the public `GEVResult.shape` follows the
standard $\xi$ convention.

Under a correct marginal CDF, the probability integral transform
$U_{t,j}=G_j(Y_{t,j})$ is uniform on $(0,1)$. The unit-Frechet transform is

$$
Z_{t,j} = -\frac{1}{\log U_{t,j}},
\qquad
\Pr(Z_{t,j}\le z)=\exp(-1/z), \quad z>0.
$$

In practice $G_j$ is estimated, so the transformed margins are not exactly
known unit-Frechet margins. The package clips probabilities near zero and one
to avoid infinities. Uncertainty in the estimated GEV parameters is not
propagated to the dependence fit.

The marginal assumption is stationary iid block maxima at each site. A GEV
fit is not a model for arbitrary raw daily observations; first define blocks
that match the scientific question, then obtain one maximum per block.

### 2.3 Dependence models

The core package provides three simple max-stable model classes:

| Model | Dependence representation | Main parameters | Key consideration |
|---|---|---|---|
| Smith | Gaussian spatial kernel with covariance matrix $\Sigma$ | Positive-definite $2\times2$ covariance | Can express anisotropy; simulation uses a finite storm-center window. |
| Schlather | Positive part of a stationary Gaussian process | Range and powered-exponential smoothness by default | Extremal coefficient is bounded above by about 1.707, so it cannot reach independence. |
| Brown-Resnick | Gaussian process with stationary increments | Range and power variogram exponent $\alpha$ by default | Custom variograms must be valid conditionally negative-definite variograms. |

All three provide unit-Frechet simulation, bivariate CDF/log-density functions,
and theoretical pairwise extremal coefficients. The model equations and
parameterizations are given in [Technical Details](./TECHNICAL_DETAILS.md).

These are **max-stable** models. They are useful for spatial dependence of
block maxima when max-stability is a reasonable approximation. A successful
fit does not establish max-stability or stationarity.

### 2.4 Pairwise composite-likelihood fitting

For site pair $(i,j)$ and replicate $t$, let
$f_{ij}(z_{t,i},z_{t,j};\psi)$ be the model's bivariate density on unit-Frechet
margins, where $\psi$ denotes dependence parameters. The package minimizes
the negative pairwise composite log-likelihood

$$
\ell_{\mathrm{pair}}(\psi)
=
\sum_{i<j}\sum_{t\in\mathcal{T}_{ij}}
\log f_{ij}(z_{t,i},z_{t,j};\psi),
$$

where $\mathcal{T}_{ij}$ includes replicates with both sites observed. For
complete data this uses all pairs and replicates; missing values are handled
pairwise. The missingness mechanism must be ignorable for the dependence
analysis.

This is a **composite likelihood**, not the full joint likelihood for all
sites. Pairwise terms are dependent because they reuse observations. The
returned `nllh` is the minimized negative composite log-likelihood; it should
not be interpreted as an ordinary full-likelihood value. The package does
not provide composite-likelihood standard errors or correct uncertainty for
the marginal estimation stage.

Typical usage:

```python
import numpy as np
from spatial_extremes import fit_pairwise_likelihood
from spatial_extremes.models import BrownResnickModel

fit = fit_pairwise_likelihood(
    BrownResnickModel,
    frechet,                         # wide table on unit-Frechet margins
    coordinates,                     # order matches frechet.columns
    x0_free=np.array([np.log(50_000), 0.0]),
    max_pairs=None,                  # or a fixed random subset for large D
    rng=np.random.default_rng(42),
)

if not fit.converged:
    raise RuntimeError(f"Optimizer did not converge: {fit.optimizer_result.message}")

print(fit.model.range_, fit.model.alpha)
```

`x0_free` is expressed in the model's unconstrained optimization
parameterization, not necessarily the model's public parameter scale. For
Brown-Resnick and Schlather, the first element is log range and the second is
a logit-like transform of the shape/smoothness parameter. Choosing a sensible
starting range in the coordinate units can help. Smith has three free
parameters for a Cholesky representation of its covariance.

For large numbers of sites, the number of pairs is $D(D-1)/2$. Setting
`max_pairs` selects a fixed subset of pairs before optimization, reducing
cost. This changes the objective and adds sampling variability; set `rng` to
make the selected subset reproducible. `fit.n_pairs_used`,
`fit.n_replicates`, `fit.converged`, and `fit.optimizer_result` help record
what happened.

### 2.5 Dependence diagnostics and interpretation

For unit-Frechet observations, transform to uniform margins using
$F(z)=\exp(-1/z)$. The F-madogram for a pair is

$$
\widehat{\nu}_F(i,j)
=
\frac{1}{2n}\sum_{t=1}^n
\left|F(z_{t,i})-F(z_{t,j})\right|.
$$

For a max-stable process, its corresponding extremal-coefficient estimate is

$$
\widehat{\theta}(i,j)
=
\frac{1+2\widehat{\nu}_F(i,j)}
     {1-2\widehat{\nu}_F(i,j)}.
$$

The package implements this through `fmadogram`,
`empirical_extremal_coefficient`, and `binned_extremal_coefficient`:

```python
from spatial_extremes import (
    binned_extremal_coefficient,
    empirical_extremal_coefficient,
    fmadogram,
)

# These diagnostics expect complete positive unit-Frechet observations.
complete = frechet.dropna()
pairs = fmadogram(complete, coordinates)
by_distance = binned_extremal_coefficient(complete, coordinates, n_bins=10)
```

Use empirical summaries to check whether the fitted model's dependence
pattern is plausible over the observed distance range. An extremal
coefficient of 1 indicates complete dependence for a pair; 2 indicates
independence. The diagnostics are descriptive and do not perform a formal
goodness-of-fit test. The implemented diagnostics do not handle missing
values themselves, hence the complete-case selection above.

In particular, Schlather's upper limit is
$1+\sqrt{2}/2\approx1.707$. If the data suggest near-independence at long
distances, a Schlather model may be inadequate. Model choice should reflect
the process and dependence diagnostics, not only which optimizer converged.

### 2.6 Optional neural Bayes estimator

Install the extra with `python -m pip install "spatial-extremes[neural]"`.
Then import it explicitly:

```python
from spatial_extremes.neural_estimator import NeuralEstimator

# NeuralEstimator requires complete unit-Frechet replicates.
neural_frechet = frechet.dropna()
estimator = NeuralEstimator(
    model="brownresnick",        # or "schlather"
    coords=coordinates,
    n_reps=len(neural_frechet),
    seed=42,
)
estimator.fit(n_train=3000, n_jobs=1)
estimate = estimator.predict(neural_frechet.to_numpy())
print(estimate)                  # [range, alpha] for Brown-Resnick
```

The estimator:

1. Draws range and shape/smoothness values from its built-in prior.
2. Simulates training datasets under those parameter values.
3. Compresses each dataset to binned dependence summaries.
4. Trains a neural network using squared-error loss to predict the
   transformed parameters (log range and shape/smoothness).

With ideal simulation and training, squared-error prediction targets the
conditional posterior mean of the transformed parameters given the summaries
under that prior. The API transforms the predicted log range back to range.
It is therefore a simulation-based **Bayes point estimator**, not a general
posterior sampler or a conventional MCMC implementation.

The network is specific to its station layout and replicate count. The
estimator assumes complete unit-Frechet input for its summary calculation;
resolve missingness appropriately before using it. Its fitted prior has
finite parameter bounds and can pull estimates toward those bounds. Use
`predict_bootstrap` only to assess sensitivity to resampling replicates; its
draws are not posterior samples or calibrated credible intervals. Compare
with pairwise likelihood on representative data and assess estimator
performance on held-out simulations before relying on it.

### 2.7 Simulation, reproducibility, and limits

Each model's `simulate` method accepts coordinates, a replicate count, and a
NumPy random generator:

```python
from spatial_extremes.models import SchlatherModel

model = SchlatherModel(range_=80_000, smoothness=1.0)
z = model.simulate(
    coordinates,
    n_reps=100,
    rng=np.random.default_rng(2026),
)
```

The result has shape `(n_reps, n_sites)` and unit-Frechet margins. Providing
an explicit generator makes repeated runs reproducible for a fixed
implementation and environment.

Simulation uses a truncated Poisson construction. Smith also restricts storm
centers to a finite window; Schlather and Brown-Resnick use probabilistic
stopping bounds. All have hard `max_points` caps. These approximations can
introduce truncation error, especially if the cap is reached. Simulation
results should be checked for the intended parameter regime and site layout.

### 2.8 Suggested analysis checklist

Before reporting results:

1. Define blocks scientifically (for example, annual maxima) and check that
   rows represent suitable independent replicates.
2. Plot each site's data over time and assess stationarity and missingness.
3. Fit and inspect marginal GEVs; note convergence and uncertainty.
4. Verify unit-Frechet transformation and site/coordinate ordering.
5. Fit a dependence model and retain optimizer status, seed, and pair count.
6. Compare empirical dependence with theoretical model dependence over the
   observed distance range.
7. Check sensitivity to reasonable model choices, starting values, and pair
   subsampling where applicable.
8. State the two-stage uncertainty limitation: marginal uncertainty is not
   propagated, and pairwise composite-likelihood standard errors are not
   supplied by this package.
9. Treat a converged fit as a parameter estimate under assumptions, not as
   proof those assumptions are valid.

For formulas and model-specific details, see [Technical Details](./TECHNICAL_DETAILS.md).
