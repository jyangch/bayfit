<p align="center">
  <strong>BayFit</strong>
</p>

<p align="center">
  <strong>Bayesian inference-based curve fitting for generic x/y data.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB?labelColor=1f2937&logo=python&logoColor=white"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img alt="License GPL-3.0" src="https://img.shields.io/badge/license-GPL--3.0-9CA3AF?labelColor=1f2937"></a>
</p>

---

`BayFit` is a Python library for Bayesian inference on generic x/y curve
data. It pairs MCMC and nested-sampling backends with multi-dataset,
multi-model fitting machinery, ships a small library of analytic model
components, and reads data from dictionaries, DataFrames, CSV, JSON, and
NumPy `.npz` archives out of the box.

It is the curve-fitting sibling of
[`BaySpec`](https://github.com/jyangch/bayspec): the same inference and
model-composition design, generalised from high-energy spectra to
arbitrary x/y curves.


## Features

- **Inference backends.** Posterior sampling via MCMC
  ([`emcee`](https://emcee.readthedocs.io/)) or nested sampling
  ([`MultiNest`](https://github.com/JohannesBuchner/MultiNest) via
  [`pymultinest`](https://johannesbuchner.github.io/PyMultiNest/));
  maximum-likelihood fits via
  [`lmfit`](https://lmfit.github.io/lmfit-py/) or
  [`iminuit`](https://iminuit.readthedocs.io/) for quick checks.
- **Multi-dataset, multi-model.** Simultaneously fit any number of
  `(data, model)` pairs; freeze or link parameters across pairs.
- **Multi-dimensional inputs.** Data carry an `(npoint, ndim)` design
  matrix, so models can depend on more than one independent variable.
- **Model algebra.** Combine components with `+`, `-`, `*`, `/` to build
  composite models from additive, multiplicative, and mathematical
  building blocks.
- **Local model library.** Linear, power-law, exponential decay,
  single/double/triple smoothly broken power-laws, magnetar spin-down,
  and power-spectral-density models, plus an exponential-cutoff factor
  and a constant — discover them with `list_local_models()`.
- **Flexible data loaders.** Build a `DataUnit` from a dict, a `pandas`
  `DataFrame`, a CSV, a JSON file, or a NumPy `.npz` archive, with
  symmetric or asymmetric x/y errors and upper/lower limits.
- **Plotting.** Plotly and Matplotlib helpers for data, models, fit
  results, and posterior corner plots.


## Installation

`BayFit` is not yet packaged for PyPI. Clone the repository and install
the dependencies, then use it from the repository root (so that
`import bayfit` resolves):

```bash
git clone https://github.com/jyangch/bayfit.git
cd bayfit
pip install numpy scipy pandas emcee matplotlib plotly corner lmfit iminuit
```

Core dependencies: `numpy`, `scipy`, `pandas`, `emcee`, `matplotlib`,
`plotly`, and `corner`. Maximum-likelihood fitting additionally uses
`lmfit` and `iminuit`.

### Optional: `MultiNest` sampler

To enable [`MultiNest`](https://github.com/JohannesBuchner/MultiNest) for
nested sampling, follow the
[`pymultinest`](https://johannesbuchner.github.io/PyMultiNest/) install
guide (it requires the `MultiNest` C library in addition to the Python
package).


## Quick start

Fit a straight line — recovering its slope, intercept, and intrinsic
scatter — to noisy data with MCMC:

```python
import numpy as np

from bayfit.data.data import Data, DataUnit
from bayfit.infer.infer import BayesInfer
from bayfit.model.local import line
from bayfit.util.plot import Plot

# synthetic data: y = 2x + 1 with measurement error *and* intrinsic scatter
rng = np.random.default_rng(1)
x = np.linspace(0, 10, 40)
yerr = np.full(x.size, 0.3)        # measurement error
sigma_int = 1.0                    # intrinsic scatter (logv_true = log10(1) = 0)
y = 2.0 * x + 1.0 + rng.normal(0, np.hypot(yerr, sigma_int))

# one data unit -> a Data container; 'chi2f' fits an extra variance term
unit = DataUnit(x, y, yerr=yerr, stat='chi2f')
data = Data([('d', unit)])

# a linear model; slope k, intercept b, and log intrinsic scatter logv are all free
model = line()

# pair data with the model and sample the posterior
infer = BayesInfer([(data, model)])
post = infer.multinest(nlive=400, resume=False, savepath='./out')

k, b, logv = post.par_best_ci[:3]
# recovers the slope, intercept, and intrinsic scatter sigma_int = 10**logv
print(f'k = {k:.2f}, b = {b:.2f}, sigma_int = {10 ** logv:.2f}')   # -> ~2.0, ~1.0, ~1.0

# inspect the fit and the posterior (auto-display inline in notebooks)
Plot.infer(post)        # data, best-fit model, and residuals
Plot.post_corner(post)  # parameter corner plot
```

Both `Plot.*` calls return a `Figure` that displays inline in notebooks.
From a script, choose a file-writing backend and save a PDF:

```python
fig = Plot.infer(post,  ploter='matplotlib')
fig.save('fit')

fig = Plot.post_corner(post, ploter='cornerpy')
fig.save('corner')
```

Swap `line()` for any component returned by `list_local_models()`, or
compose several with `+`/`*`, to fit other curves. Load real data with
`DataUnit.from_csv`, `from_json`, `from_dataframe`, or `from_npz`.


## License

`BayFit` is distributed under the
[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0-standalone.html) license.
