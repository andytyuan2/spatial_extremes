# Technical Details

This note records the formulas and assumptions used by the implementation.
It describes the models implemented in `src/spatial_extremes/models/`, not
every possible max-stable model or parameterization.

## Notation and margins

Let $Z(s)$ denote the process at a two-dimensional site $s$. The model
simulators return unit-Frechet margins:

$$
\Pr\{Z(s) \le z\} = \exp(-1/z), \qquad z > 0.
$$

For a pair separated by $h=s_2-s_1$, the bivariate distribution of a
simple max-stable process is written

$$
\Pr\{Z(s_1)\le z_1,\;Z(s_2)\le z_2\}
= \exp\{-V(z_1,z_2;h)\}.
$$

The pairwise extremal coefficient is

$$
\theta(h)=V(1,1;h), \qquad 1 \le \theta(h) \le 2.
$$

Here $\theta=1$ represents complete dependence and $\theta=2$
independence for a pair.

## GEV margins and unit-Frechet transformation

At a site, the generalized extreme-value distribution has location
$\mu$, scale $\sigma>0$, and shape $\xi$. Define

$$
t(x)=1+\xi\frac{x-\mu}{\sigma}.
$$

For $t(x)>0$ and $\xi\ne0$, its CDF is

$$
G(x)=\exp\left[-t(x)^{-1/\xi}\right].
$$

The $\xi=0$ limit is the Gumbel CDF

$$
G(x)=\exp\left[-\exp\left(-\frac{x-\mu}{\sigma}\right)\right].
$$

The corresponding log-density for $\xi\ne0$ is

$$
\log g(x)=-\log\sigma
-\left(1+\frac1\xi\right)\log t(x)-t(x)^{-1/\xi},
$$

and the Gumbel limit is

$$
\log g(x)=-\log\sigma-\frac{x-\mu}{\sigma}
-\exp\left(-\frac{x-\mu}{\sigma}\right).
$$

`fit_gev` maximizes the sum of these univariate log-densities for
independent, stationary observations. It parameterizes scale as
$\sigma=\exp(\eta)$ during optimization and uses the standard
extreme-value shape convention $\xi$. SciPy's `genextreme` uses the
opposite shape parameter, $c=-\xi$, for CDF evaluation.

Given a fitted CDF value $u=G(x)$, the unit-Frechet transform is

$$
z_F=-\frac{1}{\log u},
$$

which gives $\Pr(Z_F\le z)=\exp(-1/z)$ when $u$ is uniform. The
implementation clips $u$ away from zero and one for numerical stability.
The fitted marginal parameters are estimated, so transformed margins are
not known exactly.

## Smith model

The Smith model uses the bivariate Gaussian density

$$
\phi_\Sigma(x)
=\frac{1}{2\pi|\Sigma|^{1/2}}
\exp\left(-\frac12 x^\mathsf{T}\Sigma^{-1}x\right),
$$

where $\Sigma$ is a positive-definite $2\times2$ covariance matrix.
Its spectral representation is

$$
Z(s)=\max_i \xi_i\,\phi_\Sigma(s-U_i),
$$

with Poisson points $(\xi_i,U_i)$ having intensity
$\xi^{-2}\,d\xi\,du$. The bivariate dependence enters through the
Mahalanobis separation

$$
a(h)=\sqrt{h^\mathsf{T}\Sigma^{-1}h}.
$$

The bivariate CDF and extremal coefficient are the shared formulas in
[Shared bivariate formulas](#shared-bivariate-formulas), with this $a(h)$.
Larger covariance eigenvalues spread the Gaussian kernel farther and
generally increase dependence over a given displacement.

The simulator restricts storm centers to a finite box around the site
coordinates. Within that box, the Poisson arrival times $T_i$ are
unit-rate exponential sums and the points are scaled as
$\xi_i=A/T_i$, where $A$ is the box area. This accounts for the
spatial intensity restricted to the box; it does not restore contributions
from centers outside the box. The buffer and the `max_points` cap therefore
affect simulation accuracy.

## Schlather extremal Gaussian model

Let $\epsilon_i(s)$ be independent copies of a stationary, standard
Gaussian process with valid correlation function $\rho(h)$. The
implemented spectral representation is

$$
Z(s)=\max_i \xi_i W_i(s), \qquad
W_i(s)=\sqrt{2\pi}\,\max(0,\epsilon_i(s)).
$$

Since $E[\max(0,N)]=1/\sqrt{2\pi}$ for $N\sim N(0,1)$, the
normalization gives $E[W_i(s)]=1$ and unit-Frechet margins. The default
correlation is the powered-exponential family

$$
\rho(h)=\exp\left[-\left(\frac{\lVert h\rVert}{r}\right)^\beta\right],
\qquad r>0,\quad 0<\beta\le2.
$$

For a custom correlation function, it must be a valid stationary Gaussian
correlation: in particular, every site correlation matrix used for
simulation must be positive semidefinite. The code clips correlations
slightly inside $(-1,1)$ to avoid singular expressions and adds a small
diagonal jitter before Cholesky factorization.

For $z_1,z_2>0$, define

$$
b=\sqrt{z_1^2+z_2^2-2\rho(h)z_1z_2}, \qquad
q=1-\frac{2(1+\rho(h))z_1z_2}{(z_1+z_2)^2}.
$$

Its bivariate exponent function is

$$
V(z_1,z_2;h)
=\frac12\left(\frac1{z_1}+\frac1{z_2}\right)(1+\sqrt q),
$$

and its extremal coefficient is

$$
\theta(h)=1+\sqrt{\frac{1-\rho(h)}{2}}.
$$

As $\rho(h)\to0$, $\theta(h)\to1+1/\sqrt2\approx1.7071$, not 2.
Thus the Schlather model cannot represent asymptotic independence at
large separation.

## Brown-Resnick model

Let $\epsilon(s)$ be a mean-zero Gaussian process with stationary
increments and semivariogram

$$
\gamma(h)=\frac12\mathrm{Var}\{\epsilon(s+h)-\epsilon(s)\}.
$$

The Brown-Resnick process is represented as

$$
Z(s)=\max_i \xi_i \exp\{\epsilon_i(s)-\gamma(s-o)\},
$$

where $o$ is an arbitrary anchor and the $\epsilon_i$ are independent
copies of the Gaussian process. Because
$\mathrm{Var}(\epsilon(s)-\epsilon(o))=2\gamma(s-o)$, the
spectral weight has mean one:

$$
E[\exp\{\epsilon(s)-\gamma(s-o)\}]=1.
$$

The default power variogram is

$$
\gamma(h)=\left(\frac{\lVert h\rVert}{r}\right)^\alpha,
\qquad r>0,\quad 0<\alpha\le2.
$$

The bivariate formula uses

$$
a(h)=\sqrt{2\gamma(h)}
$$

in the shared formulas below. A custom variogram must be conditionally
negative definite and satisfy $\gamma(0)=0$ for a valid model. These
properties are not checked for a user-provided function. During
simulation, the covariance of the anchored Gaussian field is assembled
from the variogram as

$$
\mathrm{Cov}\{\epsilon(s_i),\epsilon(s_j)\}
=\gamma(s_i-o)+\gamma(s_j-o)-\gamma(s_i-s_j).
$$

## Shared bivariate formulas

For Smith and Brown-Resnick, let $a=a(h)>0$ be the model-specific
separation above, and define

$$
w=\frac a2+\frac{\log(z_2/z_1)}{a},\qquad v=a-w.
$$

Using the standard normal CDF $\Phi$, the exponent function, bivariate
CDF, and extremal coefficient are

$$
V(z_1,z_2;h)=\frac{\Phi(w)}{z_1}+\frac{\Phi(v)}{z_2},
\qquad F(z_1,z_2;h)=e^{-V(z_1,z_2;h)},
\qquad \theta(h)=2\Phi(a/2).
$$

At $a=0$, the CDF has the complete-dependence limit
$\exp\{-1/\min(z_1,z_2)\}$.

For $a>0$, set

$$
V_1=-\frac{\Phi(w)}{z_1^2},\qquad
V_2=-\frac{\Phi(v)}{z_2^2},\qquad
V_{12}=-\frac{\phi(w)}{a z_1^2 z_2},
$$

where $\phi$ is the standard normal density. The bivariate density and
log-density are

$$
f(z_1,z_2;h)=e^{-V}(V_1V_2-V_{12}),\qquad
\log f=-V+\log(V_1V_2-V_{12}).
$$

The implementation uses a small positive lower bound for $a$ in the
log-density calculation to avoid division by zero; the displayed
complete-dependence CDF limit does not imply that the density routine
implements a singular density at coincident sites.

## Pairwise composite likelihood

Let $z_{t,i}$ be the unit-Frechet observation at replicate $t$ and
site $i$, and let $\ell_{ij,t}(\vartheta)$ be the model bivariate
log-density for that site pair and replicate. The fitting routine
minimizes the negative pairwise composite log-likelihood

$$
\mathrm{NCL}(\vartheta)
=-\sum_{i<j}\sum_{t\in\mathcal{T}_{ij}}
\ell_{ij,t}(\vartheta),
$$

where $\mathcal{T}_{ij}$ contains replicates observed at both sites.
This is not the full spatial likelihood. It assumes independent
replicates and ignorable missingness, and the routine does not compute
composite-likelihood standard errors. If `max_pairs` is set, the
objective uses a fixed random subset of eligible pairs.

## Curvature-adjusted Bayesian inference

`sample_pairwise_likelihood_posterior` samples a Bayesian posterior based on
the pairwise likelihood with the local curvature adjustment of Ribatet,
Cooley & Davison (2012). Let $\hat\vartheta$ be the pairwise maximum
likelihood estimate, $H$ the observed negative Hessian of the composite
log-likelihood at that estimate, and $J$ the sum of outer products of the
replicate-level composite scores. For a replicate $t$, its score is the sum
of that replicate's score contributions over all included site pairs. The
Godambe information is

$$
G=H J^{-1}H.
$$

The adjusted log-likelihood evaluates the original composite log-likelihood
at a linearly transformed parameter,

$$
\ell_{\mathrm{adj}}(\vartheta)
=\ell\left(\hat\vartheta+C(\vartheta-\hat\vartheta)\right),
\qquad C^\mathsf{T}HC=G.
$$

The prior remains on the original free-parameter vector $\vartheta$.
Independent zero-centered normal priors with standard deviation 2.5 are
used by default; a `log_prior` callable can supply a different prior. The
posterior is sampled by random-walk Metropolis with a proposal based on
$G^{-1}$. Matrices $H$ and $J$ are estimated by finite differences and
replicate-level scores. The routine requires positive-definite estimates
and reports an error for weakly identified or insufficiently informed fits
rather than silently regularizing them. This adjustment corrects local
posterior curvature; it does not account for uncertainty in fitted margins,
temporal dependence, or model misspecification.

## Bayesian spatio-temporal GEV margins

`fit_spatiotemporal_gev` (available after installing the `bayesian` extra)
fits a hierarchical GEV model to block maxima arranged as a
time-by-site matrix. Missing observations are omitted from the marginal
likelihood. The location at time $t$ and site $s$ is

$$
\mu_{t,s}=\mu_0+\mathbf{x}_t^\mathsf{T}\boldsymbol\beta+u_{t,s},
\qquad
u_{t,s}=\phi u_{t-1,s}+\sqrt{1-\phi^2}\,\epsilon_{t,s},
$$

where the optional covariates are standardized and the innovations and
initial field have a spatial Matérn-3/2 covariance. The log-scale has a
separate static spatial Gaussian field. Shape has a static spatial field
with a strongly regularized intercept and amplitude. Each field has its
own covariance range. Sampling is performed with PyMC's NUTS sampler.
Posterior return-level draws are provided by
`BayesianGEVResult.posterior_return_levels`.

Conditional on these latent marginal fields, observations are treated as
independent. This borrows from hierarchical spatial/temporal GEV approaches
such as Cooley, Nychka & Naveau (2007) and Sang & Gelfand (2009), but it does
not include a residual max-stable or other extremal-dependence process.
Consequently, marginal posterior inference and marginal return-level maps
do not by themselves quantify joint-event risk. Probabilities of many sites
exceeding thresholds together, simultaneous flooding, and other joint
hazards may have severely understated uncertainty and should not be inferred
from this model alone. Temporal spacing is assumed equal for the AR(1)
recursion; irregularly spaced times are rejected.
The spatial covariance matrices are dense, so factorization cost grows
cubically with the number of sites.

This model follows the hierarchical marginal-inference approach of Cooley,
Nychka & Naveau, “Bayesian Spatial Modeling of Extreme Precipitation Return
Levels” (2007, *Journal of the American Statistical Association* 102(479),
824–840, [doi:10.1198/016214506000000780]), and the spatio-temporal
hierarchical perspective of Sang & Gelfand, “Hierarchical modeling for
extreme values observed over space and time” (2009, *Environmental and
Ecological Statistics* 16, 407–426,
[doi:10.1007/s10651-007-0078-0]).

## F-madogram diagnostic

For unit-Frechet observations, transform $U(s)=\exp\{-1/Z(s)\}$, which
is uniform on $[0,1]$ marginally. The F-madogram is

$$
\nu_F(h)=\frac12 E\left[|U(s)-U(s+h)|\right].
$$

For a simple max-stable pair, its population relation to the extremal
coefficient is

$$
\theta(h)=\frac{1+2\nu_F(h)}{1-2\nu_F(h)}.
$$

The package computes the empirical mean over complete paired rows.
These diagnostics are summaries and are not goodness-of-fit tests.

## Workflow

```text
Block maxima
     |
     v
Fit a stationary GEV at each site
     |
     v
Transform fitted CDF values to unit-Frechet
     |
     +----> Empirical F-madogram / extremal-coefficient summaries
     |
     v
Fit a stationary spatial model by pairwise composite likelihood
```

The workflow is two-step: marginal parameter uncertainty is not
propagated into the spatial fit. Trends, covariate-dependent margins,
non-stationary dependence, and temporal dependence between replicates
require additional modeling not provided here.
