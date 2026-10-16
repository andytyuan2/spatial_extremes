# spatial-extremes

Python tools for simulating and fitting max-stable processes (Smith,
Schlather, Brown-Resnick), plus GEV marginal fitting and spatial
dependence diagnostics -- a from-scratch Python port of the ideas
behind R's `SpatialExtremes` package.

## Install

```bash
pip install -e ".[geo,polars,dev]"
```

## What's here

- `spatial_extremes.gev` -- univariate GEV fitting via MLE (`fit_gev`,
  `fit_gev_by_site`), unit-Frechet margin transform.
- `spatial_extremes.models` -- `SmithModel`, `SchlatherModel`,
  `BrownResnickModel`: simulation (`.simulate`), closed-form bivariate
  CDF/density (`.bivariate_cdf`, `.bivariate_logpdf`), and the
  theoretical extremal coefficient (`.extremal_coefficient`).
- `spatial_extremes.dependence` -- model-free empirical diagnostics
  (F-madogram, empirical/binned extremal coefficient) for sanity-checking
  a fitted model against data.
- `spatial_extremes.fitting` -- `fit_pairwise_likelihood`: generic
  composite-likelihood parameter fitting that works with any of the
  three models.

## Status / honesty notes

- Every model's simulator is validated in `tests/` against its
  closed-form bivariate CDF (Monte Carlo agreement), not just "runs
  without crashing."
- `fit_pairwise_likelihood` is validated end-to-end: simulate from a
  known model, fit blind, check the fitted extremal-coefficient
  function matches the true one (see `tests/test_fitting.py`).
- Schlather's model has a known, real limitation baked into the math
  (not a bug here): its extremal coefficient saturates around 1.707 and
  can never represent full independence between sites, however far
  apart.
- No code was ported from R's `SpatialExtremes` (GPL); all formulas
  here were derived from the published literature (Smith 1990;
  Schlather 2002; Kabluchko, Schlather & de Haan 2009; Padoan, Ribatet
  & Sisson 2010) and cross-checked symbolically with sympy against
  finite differences.
- Geospatial integration uses `geopandas` (production-ready). `polars`
  is supported for tabular data, but there's currently no
  production-ready polars-native geospatial library (GeoPolars is
  still a prototype; `polars-st` -- LGPL, GEOS-backed -- is the closest
  practical option if you want to explore that route yourself).

## Not yet built

- Marginal (GEV) parameters aren't yet folded into the pairwise
  likelihood -- currently you fit GEV per site, transform to unit
  Frechet, then fit spatial dependence separately (the standard
  two-step approach, but a joint model is possible as a next step).
- No conditional simulation, no non-stationary variogram parametrizations.
