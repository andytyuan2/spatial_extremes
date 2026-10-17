"""
Amortized ("neural Bayes") parameter estimator for max-stable spatial models.

IDEA
----
Pairwise-likelihood fitting (fitting.py) re-evaluates ~D^2/2 bivariate
densities at every optimizer step, for every dataset. Here we instead:

  1. Draw parameters theta from a prior.
  2. Simulate a dataset Z ~ model(theta) with the repo's own simulator.
  3. Compress Z to a small vector of summary statistics s(Z)
     (binned F-madogram + binned log-madogram, i.e. empirical dependence
     curves versus distance).
  4. Train a neural network  s(Z) -> theta  with squared-error loss.

Squared-error loss makes the network approximate the *posterior mean*
E[theta | s(Z)] under your prior (a "neural Bayes estimator", cf. Lenzi
et al. 2023; Sainsbury-Dale, Zammit-Mangion & Huser 2024). After the
one-off training cost, estimating a new dataset is a single forward
pass (milliseconds), and bootstrapping the replicates gives cheap
uncertainty estimates.

SUPPORTED MODELS: "brownresnick" -> (range, alpha), "schlather" -> (range,
smoothness). Both isotropic. (Smith's anisotropic covariance can't be
identified from distance-only summaries; isotropic Smith is Brown-Resnick
with alpha = 2.)

USAGE (real data)
-----------------
    from spatial_extremes import fit_gev
    # annual maxima: DataFrame, rows = years, columns = stations
    est = NeuralEstimator("brownresnick", coords=station_xy, n_reps=len(maxima))
    est.fit(n_train=3000, n_jobs=-1)
    est.save("nbe_br.joblib")

    z = np.column_stack([fit_gev(maxima[c]).to_unit_frechet(maxima[c])   # unit-Frechet margins
                         for c in maxima.columns])
    print(est.predict(z))             # [range, alpha]
    print(est.predict_bootstrap(z))   # (200, 2) bootstrap draws

CAVEATS
-------
* The network is trained for ONE station layout and ONE replicate count
  (n_reps); retrain if either changes. Coordinates must be projected
  (same linear units everywhere), not lat/lon degrees.
* Training data have exactly unit-Frechet margins. Real data have
  *estimated* margins, so expect a bit of extra error that this model
  does not know about. Check with a pairwise-likelihood fit on a few sites.
* The estimate is a posterior mean under the prior in `_prior`; if the
  truth is near a prior boundary the estimate is pulled inward.
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import joblib
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# The package directory in the repo is `src/spatial-extremes` (hyphen), which
# Python cannot import; rename it to `spatial_extremes` (see packaging notes).
from spatial_extremes import BrownResnickModel, SchlatherModel, fit_pairwise_likelihood

_MODELS = {
    "brownresnick": (BrownResnickModel, "alpha", (0.2, 1.9)),
    "schlather": (SchlatherModel, "smoothness", (0.2, 2.0)),
}


def _build(model: str, range_: float, shape: float):
    cls, shape_name, _ = _MODELS[model]
    return cls(range_=float(range_), **{shape_name: float(shape)})


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------
@dataclass
class Features:
    """Distance-binned dependence summaries for a fixed station layout.

    Bin edges are quantiles of the pairwise distances, so every bin holds
    (roughly) the same number of site pairs and none is ever empty.
    """

    coords: np.ndarray
    n_bins: int = 12

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        D = len(self.coords)
        self.i, self.j = np.triu_indices(D, k=1)
        d = np.linalg.norm(self.coords[self.i] - self.coords[self.j], axis=1)
        edges = np.unique(np.quantile(d, np.linspace(0, 1, self.n_bins + 1)))
        self.n_bins = len(edges) - 1            # ties (e.g. grids) can merge bins
        edges[-1] += 1e-9
        self.bin_id = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, self.n_bins - 1)
        self.counts = np.bincount(self.bin_id, minlength=self.n_bins).astype(float)
        self.mean_dist = np.bincount(self.bin_id, weights=d, minlength=self.n_bins) / self.counts

    def _bin_mean(self, per_pair):
        return np.bincount(self.bin_id, weights=per_pair, minlength=self.n_bins) / self.counts

    def __call__(self, z) -> np.ndarray:
        """z: (n_reps, D) unit-Frechet data -> (2 * n_bins,) feature vector."""
        z = np.maximum(np.asarray(z, dtype=float), 1e-12)
        u = np.exp(-1.0 / z)                                    # uniform margins
        nu = 0.5 * np.abs(u[:, self.i] - u[:, self.j]).mean(axis=0)   # F-madogram
        lz = np.log(z)
        lm = np.abs(lz[:, self.i] - lz[:, self.j]).mean(axis=0)       # log-madogram
        return np.concatenate([self._bin_mean(nu), self._bin_mean(lm)])


def _simulate_features(model, params, coords, n_reps, feats, seed):
    rng = np.random.default_rng(seed)
    z = _build(model, *params).simulate(coords, n_reps=n_reps, rng=rng)
    return feats(z)


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------
class NeuralEstimator:
    def __init__(self, model="brownresnick", coords=None, n_reps=30, n_bins=12,
                 hidden=(128, 128, 64), seed=0):
        if model not in _MODELS:
            raise ValueError(f"model must be one of {list(_MODELS)}")
        if coords is None:
            raise ValueError("coords (D, 2) are required: the network is layout-specific")
        self.model, self.n_reps, self.seed = model, int(n_reps), seed
        self.coords = np.asarray(coords, dtype=float)
        self.feats = Features(self.coords, n_bins)
        self.L = float(np.linalg.norm(self.coords[self.feats.i] - self.coords[self.feats.j], axis=1).max())
        self.shape_lo, self.shape_hi = _MODELS[model][2]
        self.net = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", alpha=1e-4,
                                early_stopping=True, validation_fraction=0.1,
                                n_iter_no_change=25, max_iter=800, random_state=seed)
        self.sx, self.sy = StandardScaler(), StandardScaler()
        self.fitted = False

    # ---- prior ----------------------------------------------------------
    def _prior(self, rng, n):
        """range ~ logUniform(0.05 L, 2 L);  shape ~ Uniform(shape_lo, shape_hi)."""
        log_r = rng.uniform(np.log(0.05 * self.L), np.log(2.0 * self.L), n)
        shape = rng.uniform(self.shape_lo, self.shape_hi, n)
        return np.exp(log_r), shape

    # ---- training data / fitting ----------------------------------------
    def simulate_training_set(self, n, n_jobs=1, seed=None):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        r, s = self._prior(rng, n)
        seeds = rng.integers(0, 2**63 - 1, n)
        X = Parallel(n_jobs=n_jobs)(
            delayed(_simulate_features)(self.model, (r[k], s[k]), self.coords,
                                        self.n_reps, self.feats, int(seeds[k]))
            for k in range(n))
        return np.asarray(X), np.column_stack([np.log(r), s])    # targets: (log range, shape)

    def fit(self, n_train=2000, n_jobs=1, verbose=True):
        t0 = time.time()
        X, Y = self.simulate_training_set(n_train, n_jobs=n_jobs, seed=self.seed)
        if verbose:
            print(f"simulated {n_train} datasets in {time.time() - t0:.0f}s; training network...")
        self.net.fit(self.sx.fit_transform(X), self.sy.fit_transform(Y))
        self.fitted = True
        if verbose:
            print(f"done. epochs={self.net.n_iter_}, best val score (R^2)={self.net.best_validation_score_:.3f}")
        return self

    # ---- inference --------------------------------------------------------
    def _predict_features(self, X):
        Y = self.sy.inverse_transform(self.net.predict(self.sx.transform(np.atleast_2d(X))))
        r = np.exp(np.clip(Y[:, 0], np.log(0.05 * self.L), np.log(2.0 * self.L)))
        s = np.clip(Y[:, 1], self.shape_lo, self.shape_hi)
        return np.column_stack([r, s])

    def predict(self, frechet) -> np.ndarray:
        """frechet: (n_reps, D) array or DataFrame on unit-Frechet margins -> [range, shape]."""
        assert self.fitted, "call fit() or load() first"
        z = np.asarray(frechet, dtype=float)
        return self._predict_features(self.feats(z))[0]

    def predict_bootstrap(self, frechet, n_boot=200, seed=0) -> np.ndarray:
        """Resample replicates (rows) with replacement; returns (n_boot, 2) draws."""
        z = np.asarray(frechet, dtype=float)
        rng = np.random.default_rng(seed)
        X = np.array([self.feats(z[rng.integers(0, len(z), len(z))]) for _ in range(n_boot)])
        return self._predict_features(X)

    def extremal_coefficient(self, params, h) -> float:
        return _build(self.model, *params).extremal_coefficient(np.array([h, 0.0]))

    # ---- evaluation vs. the repo's pairwise likelihood -----------------
    def evaluate(self, n_test=200, n_compare=8, verbose=True):
        """Held-out check on prior draws. Compares parameter error and the error in
        theta(h) at the median site distance (the quantity data actually identify)
        for the network and for fit_pairwise_likelihood on the first n_compare sets."""
        rng = np.random.default_rng(self.seed + 999)
        r, s = self._prior(rng, n_test)
        seeds = rng.integers(0, 2**63 - 1, n_test)
        data = [_build(self.model, r[k], s[k]).simulate(
                    self.coords, n_reps=self.n_reps, rng=np.random.default_rng(int(seeds[k])))
                for k in range(n_test)]
        h_med = float(np.median(np.linalg.norm(
            self.coords[self.feats.i] - self.coords[self.feats.j], axis=1)))

        t0 = time.time()
        est = self._predict_features(np.array([self.feats(z) for z in data]))
        t_nn = (time.time() - t0) / n_test
        truth = np.column_stack([r, s])
        th_true = np.array([self.extremal_coefficient(p, h_med) for p in truth])
        th_nn = np.array([self.extremal_coefficient(p, h_med) for p in est])

        def rep(name, e, th):
            return {"method": name,
                    "RMSE log(range)": np.sqrt(np.mean((np.log(e[:, 0]) - np.log(truth[:len(e), 0])) ** 2)),
                    "RMSE shape": np.sqrt(np.mean((e[:, 1] - truth[:len(e), 1]) ** 2)),
                    "MAE theta(h_med)": np.mean(np.abs(th - th_true[:len(th)]))}

        rows = [rep(f"neural (all {n_test})", est, th_nn)]
        out = {"seconds_per_fit_nn": t_nn}
        if n_compare:
            cls = _MODELS[self.model]
            from spatial_extremes import BrownResnickModel as BR
            from spatial_extremes import SchlatherModel as SC
            mcls = BR if self.model == "brownresnick" else SC
            est_pl, t0 = [], time.time()
            for z in data[:n_compare]:
                res = fit_pairwise_likelihood(mcls, z, self.coords,
                                              x0_free=np.array([np.log(self.L / 4), 0.0]))
                m = res.model
                est_pl.append([m.range_, getattr(m, cls[1])])
            out["seconds_per_fit_pl"] = (time.time() - t0) / n_compare
            est_pl = np.array(est_pl)
            th_pl = np.array([self.extremal_coefficient(p, h_med) for p in est_pl])
            rows.append(rep(f"neural (same {n_compare})", est[:n_compare], th_nn[:n_compare]))
            rows.append(rep(f"pairwise lik. (same {n_compare})", est_pl, th_pl))
        out["table"] = pl.DataFrame(rows)
        if verbose:
            print(out["table"].with_columns(
                pl.exclude("method").round(3)
            ))
            print(f"time per dataset: neural {t_nn * 1e3:.2f} ms" +
                  (f", pairwise likelihood {out['seconds_per_fit_pl']:.1f} s" if n_compare else ""))
        return out

    # ---- persistence --------------------------------------------------------
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> NeuralEstimator:
        return joblib.load(path)


# --------------------------------------------------------------------------
# CLI:  python neural_estimator.py --model brownresnick --n-train 2000
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0])
    ap.add_argument("--model", default="brownresnick", choices=list(_MODELS))
    ap.add_argument("--coords-csv", help="CSV with columns x,y (projected units). Default: 20 random sites in [0,10]^2")
    ap.add_argument("--n-reps", type=int, default=30, help="replicates (e.g. years) per dataset")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--n-compare", type=int, default=8, help="# test sets also fit by pairwise likelihood (0 = skip)")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default="nbe.joblib")
    a = ap.parse_args()

    if a.coords_csv:
        with open(a.coords_csv, newline="", encoding="utf-8") as coords_file:
            reader = csv.DictReader(coords_file)
            if reader.fieldnames is None or not {"x", "y"}.issubset(reader.fieldnames):
                raise ValueError("coords CSV must contain 'x' and 'y' columns")
            c = np.array([[float(row["x"]), float(row["y"])] for row in reader])
    else:
        c = np.random.default_rng(1).uniform(0, 10, (20, 2))

    est = NeuralEstimator(a.model, coords=c, n_reps=a.n_reps).fit(a.n_train, n_jobs=a.n_jobs)
    est.evaluate(a.n_test, a.n_compare)
    est.save(a.out)
    print("saved", a.out)

    # demo: recover known parameters from one fresh dataset, with bootstrap spread
    true = (0.4 * est.L, 1.0)
    z = _build(a.model, *true).simulate(c, n_reps=a.n_reps, rng=np.random.default_rng(123))
    boot = est.predict_bootstrap(z)
    print(f"\ndemo  truth={np.round(true, 2)}  estimate={np.round(est.predict(z), 2)}  "
          f"bootstrap 90% interval: range [{np.quantile(boot[:, 0], .05):.2f}, {np.quantile(boot[:, 0], .95):.2f}], "
          f"shape [{np.quantile(boot[:, 1], .05):.2f}, {np.quantile(boot[:, 1], .95):.2f}]")
