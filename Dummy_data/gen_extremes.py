import numpy as np
import polars as pl
from scipy.spatial.distance import cdist
from scipy.stats import genextreme, norm

rng = np.random.default_rng(42)
n_sites, n_years, start = 200, 50, 1975

# Stations over a lon/lat box (roughly southern Ontario)
lon = rng.uniform(-82, -76, n_sites)
lat = rng.uniform(42, 46, n_sites)
elev = np.clip(200 + 60*(lat-42) + rng.normal(0, 50, n_sites), 0, None)

# Covariate-dependent GEV parameters (true values, kept for validation)
mu    = 30 + 4*(lat-44) - 0.02*(elev-300) + 2*np.sin(lon)          # location (mm)
sigma = np.exp(np.log(8) + 0.1*(lat-44))                           # scale
xi    = np.full(n_sites, 0.1) + 0.03*(lat-44)                      # shape
trend = 0.05                                                       # mu change per year (mm)

# Spatial dependence: Gaussian copula, exponential covariance (~150 km range)
xy_km = np.c_[lon*111*np.cos(np.radians(44)), lat*111]
C = np.exp(-cdist(xy_km, xy_km) / 150) + 1e-8*np.eye(n_sites)
L = np.linalg.cholesky(C)
Z = L @ rng.standard_normal((n_sites, n_years))     # sites x years, independent in time
U = norm.cdf(Z)

years = np.arange(start, start + n_years)
mu_t = mu[:, None] + trend*(years[None, :] - years.mean())
# scipy genextreme uses c = -xi
X = genextreme.ppf(U, c=-xi[:, None], loc=mu_t, scale=sigma[:, None])

stations = pl.DataFrame({
    "station_id": [f"S{i:03d}" for i in range(n_sites)],
    "lon": lon, "lat": lat, "elevation_m": elev,
    "true_mu": mu, "true_sigma": sigma, "true_xi": xi, "true_trend": trend,
})
annual_max = X.ravel().copy()
annual_max[rng.random(annual_max.size) < 0.02] = np.nan
obs = pl.DataFrame({
    "station_id": np.repeat(stations.get_column("station_id").to_numpy(), n_years),
    "year": np.tile(years, n_sites),
    "lon": np.repeat(lon, n_years), "lat": np.repeat(lat, n_years),
    "elevation_m": np.repeat(elev, n_years),
    "annual_max_mm": annual_max,
})
obs.write_parquet("extremes_observations.parquet")
stations.write_parquet("extremes_stations_truth.parquet")
print(obs.shape, {
    "count": int(np.isfinite(annual_max).sum()),
    "mean": round(float(np.nanmean(annual_max)), 2),
    "std": round(float(np.nanstd(annual_max)), 2),
})
