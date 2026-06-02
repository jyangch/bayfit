# infer (stat, residual) + MaxLikeFit Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align curvefit's infer layer with bayspec: every statistic returns `(stat, residual)` via numba `_*_core` kernels behind a unified `**kwargs` interface; `Pair`/`Infer` gain a `pseudo_residual` flow and `calc_*` entry points; `posterior.py` becomes `SampleAnalyzer` + `Posterior` + `Bootstrap`; and a new `MaxLikeFit(Infer)` provides `lmfit()`/`iminuit()` maximum-likelihood drivers with covariance bootstrap.

**Architecture:** Each `Statistic.Name(**kwargs)` evaluates the model (`my = mo_func(x, params)`), guards non-finite, and delegates to a module-level core returning `(stat, residual)` where `stat = ΣSᵢ = −2·loglike` and `residualᵢ = sign(y−myᵢ)·√max(Sᵢ,0)`. The five Gaussian-type cores are `@nb.njit` explicit loops; `_groth_core` stays pure Python (Decimal). lmfit minimizes the concatenated `pseudo_residual` (restricted to `chi^2`/`logchi^2`); iminuit minimizes the scalar `stat` (all six). Both bootstrap a covariance sample and return a `Bootstrap`.

**Tech Stack:** Python 3.12, numpy, numba 0.62, scipy, lmfit 1.3, iminuit 2.30, pytest, ruff.

**Conventions (do not violate):**
- NEVER modify the sibling `bayspec` package (read-only reference: `/Users/junyang/Documents/python_works/bayspec`).
- Match bayspec style + `docs/docstring-standard.md`: Google-style docstrings (English, ASCII punctuation, **no PEP 484 hints** — legacy §7), `OrderedDict`, one blank line between methods.
- The six statistic **formulas are unchanged**; only the return contract (`(stat, residual)`) and the loop form change.
- `Par.val` is the settable raw value; `Par.value` is read-only. Tests set `.val`.
- After each task run `ruff check . && ruff format --check . && python -m pytest -q`; commit only when all pass. Run pytest as its own command (don't pipe through `tail`).
- numba's first call compiles (a few seconds) — this is normal.

---

### Task 1: Rewrite `infer/statistic.py` — `(stat, residual)` + njit cores + `**kwargs`

**Files:**
- Replace: `curvefit/infer/statistic.py`
- Test: `tests/test_statistic_residual.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_statistic_residual.py`:
```python
import numpy as np

from curvefit.infer.statistic import LMFIT_SAFE_STATS, Statistic


def lin(x, params):
    return params[0] * np.asarray(x) + params[1]


def _kwargs(stat, params, up=None):
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.1, 2.1, 3.9, 6.2])
    n = x.shape[0]
    yerr = np.vstack([np.full(n, 0.2), np.full(n, 0.2)])
    xerr = np.vstack([np.full(n, 0.1), np.full(n, 0.1)])
    w = np.ones(n)
    up = np.zeros(n, dtype=bool) if up is None else np.asarray(up, dtype=bool)
    return dict(mo_func=lin, params=params, x=x, y=y, x_err=xerr, y_err=yerr, w=w, up=up)


def test_chi_square_returns_stat_and_residual():
    out = Statistic.chi_square(**_kwargs('chi^2', [2.0, 0.0]))
    assert isinstance(out, tuple) and len(out) == 2
    stat, residual = out
    # chi^2: residual is the signed sqrt of each term, so sum(res**2) == stat
    assert np.isclose(np.sum(residual**2), stat)
    # and equals weighted (y - m)/yerr
    kw = _kwargs('chi^2', [2.0, 0.0])
    my = lin(kw['x'], [2.0, 0.0])
    expected = np.sign(kw['y'] - my) * np.abs(kw['y'] - my) / 0.2
    assert np.allclose(residual, expected)


def test_stat_equals_minus_two_loglike():
    # chi^2 stat must be sum of (y-m)^2/yerr^2
    kw = _kwargs('chi^2', [2.0, 0.0])
    stat, _ = Statistic.chi_square(**kw)
    my = lin(kw['x'], [2.0, 0.0])
    assert np.isclose(stat, np.sum((kw['y'] - my) ** 2 / 0.2**2))


def test_all_six_return_tuple_finite():
    cases = {
        'chi^2': [2.0, 0.0],
        'chi^2f': [2.0, 0.0, -1.0],
        'logchi^2': [2.0, 0.1],
        'vdr': [2.0, 0.0, -1.0],
        'odr': [2.0, 0.0, -1.0],
        'groth': [2.0, 0.0],
    }
    for name, params in cases.items():
        func = Statistic._allowed_stats[name]
        stat, residual = func(**_kwargs(name, params))
        assert np.isfinite(stat), name
        assert residual.shape[0] == 4, name


def test_upper_limit_residual_behavior():
    # one upper-limit point; model below y -> allowed -> residual 0, stat finite
    kw = _kwargs('chi^2', [2.0, 0.0], up=[False, False, False, True])
    # at x=3, model=6.0, y=6.2 -> y>=my allowed -> contribution 0
    stat, residual = Statistic.chi_square(**kw)
    assert residual[3] == 0.0
    assert np.isfinite(stat)


def test_nonfinite_model_returns_inf():
    def bad(x, params):
        return np.full(np.shape(x), np.nan)

    kw = _kwargs('chi^2', [2.0, 0.0])
    kw['mo_func'] = bad
    stat, residual = Statistic.chi_square(**kw)
    assert stat == np.inf
    assert np.all(np.isinf(residual))


def test_lmfit_safe_set():
    assert LMFIT_SAFE_STATS == frozenset({'chi^2', 'logchi^2'})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_statistic_residual.py -v`
Expected: FAIL — current stats return a scalar loglike, not `(stat, residual)`; `LMFIT_SAFE_STATS` undefined.

- [ ] **Step 3: Replace the file**

Overwrite `curvefit/infer/statistic.py` with:
```python
"""Likelihood-statistic kernels for generic curve fitting.

Each statistic evaluates the model on the data x-grid, forms the per-point
``-2 * loglike`` contribution ``S``, and returns ``(stat, residual)`` where
``stat = sum(S)`` and ``residual`` is the signed square-root of each ``S``
(so lmfit's least-squares minimises the same objective for pure chi-square
statistics). Upper-limit points contribute 0 when consistent and ``inf``
otherwise.

The Gaussian-family cores (``chi^2``, ``chi^2f``, ``logchi^2``, ``vdr``,
``odr``) are numba-accelerated explicit loops; ``groth`` uses ``Decimal``
and stays pure Python. Every statistic shares one keyword bundle:
``mo_func, params, x, y, x_err, y_err, w, up``.

Attributes:
    LMFIT_SAFE_STATS: Statistic tags whose residual is a true signed
        chi-residual, so lmfit least-squares is exact for them.
"""

from decimal import Decimal
from math import factorial
from types import MappingProxyType

import numba as nb
import numpy as np

LMFIT_SAFE_STATS = frozenset({'chi^2', 'logchi^2'})


@nb.njit(cache=True, fastmath=True)
def _chi_square_core(my, y, y_err, w, up):
    """Numba kernel for ``chi^2``; returns ``(stat, signed_residual)``."""

    n = y.shape[0]
    residual = np.empty(n, dtype=np.float64)
    stat = 0.0

    for i in range(n):
        yi = y[i]
        myi = my[i]

        yerr = y_err[1, i] if yi < myi else y_err[0, i]
        si = w[i] * (yi - myi) ** 2 / yerr**2

        if up[i]:
            si = 0.0 if yi >= myi else np.inf

        stat += si

        sign = 1.0 if yi > myi else (-1.0 if yi < myi else 0.0)
        residual[i] = sign * np.sqrt(si if si > 0.0 else 0.0)

    return stat, residual


@nb.njit(cache=True, fastmath=True)
def _chi_square_full_core(my, y, y_err, w, up, logv):
    """Numba kernel for ``chi^2f`` (extra log-variance ``logv``)."""

    n = y.shape[0]
    residual = np.empty(n, dtype=np.float64)
    stat = 0.0

    for i in range(n):
        yi = y[i]
        myi = my[i]

        yerr = y_err[1, i] if yi < myi else y_err[0, i]
        sigma2 = yerr**2 + 10.0 ** (2.0 * logv)
        si = w[i] * ((yi - myi) ** 2 / sigma2 + np.log(2.0 * np.pi * sigma2))

        if up[i]:
            si = 0.0 if yi >= myi else np.inf

        stat += si

        sign = 1.0 if yi > myi else (-1.0 if yi < myi else 0.0)
        residual[i] = sign * np.sqrt(si if si > 0.0 else 0.0)

    return stat, residual


@nb.njit(cache=True, fastmath=True)
def _log_chi_square_core(my, y, y_err, w, up):
    """Numba kernel for ``logchi^2`` (chi-square in log10 space)."""

    n = y.shape[0]
    residual = np.empty(n, dtype=np.float64)
    stat = 0.0

    for i in range(n):
        logy = np.log10(y[i])
        logmy = np.log10(my[i])

        if logy < logmy:
            logyerr = np.log10(y[i] + y_err[1, i]) - logy
        else:
            logyerr = logy - np.log10(y[i] - y_err[0, i])

        si = w[i] * (logy - logmy) ** 2 / logyerr**2

        if up[i]:
            si = 0.0 if logy >= logmy else np.inf

        stat += si

        sign = 1.0 if logy > logmy else (-1.0 if logy < logmy else 0.0)
        residual[i] = sign * np.sqrt(si if si > 0.0 else 0.0)

    return stat, residual


@nb.njit(cache=True, fastmath=True)
def _vdr_core(my, x, y, x_err, y_err, w, up, k, logv):
    """Numba kernel for ``vdr`` (effective-variance regression, linear model)."""

    n = y.shape[0]
    residual = np.empty(n, dtype=np.float64)
    stat = 0.0

    for i in range(n):
        yi = y[i]
        myi = my[i]

        yerr = y_err[1, i] if yi < myi else y_err[0, i]
        if k > 0:
            xerr = x_err[0, i] if yi < myi else x_err[1, i]
        else:
            xerr = x_err[1, i] if yi < myi else x_err[0, i]

        sigma2 = np.exp(2.0 * logv) + k**2 * xerr**2 + yerr**2
        si = w[i] * ((yi - myi) ** 2 / sigma2 + np.log(2.0 * np.pi * sigma2))

        if up[i]:
            si = 0.0 if yi >= myi else np.inf

        stat += si

        sign = 1.0 if yi > myi else (-1.0 if yi < myi else 0.0)
        residual[i] = sign * np.sqrt(si if si > 0.0 else 0.0)

    return stat, residual


@nb.njit(cache=True, fastmath=True)
def _odr_core(my, x, y, x_err, y_err, w, up, k, logv):
    """Numba kernel for ``odr`` (orthogonal-distance regression, linear model)."""

    n = y.shape[0]
    residual = np.empty(n, dtype=np.float64)
    stat = 0.0

    for i in range(n):
        yi = y[i]
        myi = my[i]

        yerr = y_err[1, i] if yi < myi else y_err[0, i]
        if k > 0:
            xerr = x_err[0, i] if yi < myi else x_err[1, i]
        else:
            xerr = x_err[1, i] if yi < myi else x_err[0, i]

        delta2 = (yi - myi) ** 2 / (k**2 + 1.0)
        sigma2 = (k**2 * xerr**2 + yerr**2) / (k**2 + 1.0) + np.exp(2.0 * logv)
        si = w[i] * (delta2 / sigma2 + np.log(2.0 * np.pi * sigma2))

        if up[i]:
            si = 0.0 if yi >= myi else np.inf

        stat += si

        sign = 1.0 if yi > myi else (-1.0 if yi < myi else 0.0)
        residual[i] = sign * np.sqrt(si if si > 0.0 else 0.0)

    return stat, residual


def _groth_core(my, y):
    """Pure-Python kernel for the Groth exact-Poisson statistic.

    Sums the per-point exact Poisson log-likelihood via a ``Decimal``
    series (Groth 1975); returns ``(stat, residual)`` with
    ``stat = -2 * sum(lnL)``.
    """

    n = y.shape[0]
    residual = np.empty(n, dtype=np.float64)
    stat = 0.0

    for i in range(n):
        yi = Decimal(float(y[i]))
        myi = Decimal(float(my[i]))

        m = 0
        dft = 0
        while True:
            dfti = (yi**m) * (myi**m) / (factorial(m)) ** 2
            if dfti > 1e-20:
                dft += dfti
                m += 1
            else:
                break

        li = np.exp(-(float(yi) + float(myi))) * float(dft)
        si = -2.0 * np.log(li)
        stat += si

        sign = 1.0 if y[i] > my[i] else (-1.0 if y[i] < my[i] else 0.0)
        residual[i] = sign * np.sqrt(si if si > 0.0 else 0.0)

    return stat, residual


class Statistic:
    """Curve-fit statistic dispatch table returning ``(stat, residual)``.

    Every method takes the shared keyword bundle ``mo_func, params, x, y,
    x_err, y_err, w, up``, evaluates ``my = mo_func(x, params)``, guards
    against a non-finite model, and delegates to the matching kernel.
    ``stat`` is the summed ``-2 * loglike``; ``residual`` is the signed
    per-point square-root used by lmfit.

    Attributes:
        _allowed_stats: Read-only mapping from statistic tag to method.
    """

    @staticmethod
    def chi_square(**kwargs):
        """Gaussian chi-square with fixed per-point errors."""

        x = np.asarray(kwargs['x'], dtype=float)
        y = np.asarray(kwargs['y'], dtype=float)
        y_err = np.asarray(kwargs['y_err'], dtype=float)
        w = np.asarray(kwargs['w'], dtype=float)
        up = np.asarray(kwargs['up'], dtype=np.bool_)

        my = np.asarray(kwargs['mo_func'](x, kwargs['params']), dtype=float)
        if not np.isfinite(my).all():
            return np.inf, np.full(y.shape[0], np.inf)

        return _chi_square_core(my, y, y_err, w, up)

    @staticmethod
    def chi_square_full(**kwargs):
        """Gaussian chi-square with an extra fitted log-variance ``logv``."""

        params = kwargs['params']
        x = np.asarray(kwargs['x'], dtype=float)
        y = np.asarray(kwargs['y'], dtype=float)
        y_err = np.asarray(kwargs['y_err'], dtype=float)
        w = np.asarray(kwargs['w'], dtype=float)
        up = np.asarray(kwargs['up'], dtype=np.bool_)

        my = np.asarray(kwargs['mo_func'](x, params[:-1]), dtype=float)
        if not np.isfinite(my).all():
            return np.inf, np.full(y.shape[0], np.inf)

        return _chi_square_full_core(my, y, y_err, w, up, float(params[-1]))

    @staticmethod
    def log_chi_square(**kwargs):
        """Chi-square evaluated in log10 space."""

        x = np.asarray(kwargs['x'], dtype=float)
        y = np.asarray(kwargs['y'], dtype=float)
        y_err = np.asarray(kwargs['y_err'], dtype=float)
        w = np.asarray(kwargs['w'], dtype=float)
        up = np.asarray(kwargs['up'], dtype=np.bool_)

        my = np.asarray(kwargs['mo_func'](x, kwargs['params']), dtype=float)
        if not np.isfinite(my).all():
            return np.inf, np.full(y.shape[0], np.inf)

        return _log_chi_square_core(my, y, y_err, w, up)

    @staticmethod
    def vdr(**kwargs):
        """Effective-variance regression for a linear model ``[k, b, logv]``."""

        params = kwargs['params']
        x = np.asarray(kwargs['x'], dtype=float)
        y = np.asarray(kwargs['y'], dtype=float)
        x_err = np.asarray(kwargs['x_err'], dtype=float)
        y_err = np.asarray(kwargs['y_err'], dtype=float)
        w = np.asarray(kwargs['w'], dtype=float)
        up = np.asarray(kwargs['up'], dtype=np.bool_)

        my = np.asarray(kwargs['mo_func'](x, params), dtype=float)
        if not np.isfinite(my).all():
            return np.inf, np.full(y.shape[0], np.inf)

        return _vdr_core(my, x, y, x_err, y_err, w, up, float(params[0]), float(params[2]))

    @staticmethod
    def odr(**kwargs):
        """Orthogonal-distance regression for a linear model ``[k, b, logv]``."""

        params = kwargs['params']
        x = np.asarray(kwargs['x'], dtype=float)
        y = np.asarray(kwargs['y'], dtype=float)
        x_err = np.asarray(kwargs['x_err'], dtype=float)
        y_err = np.asarray(kwargs['y_err'], dtype=float)
        w = np.asarray(kwargs['w'], dtype=float)
        up = np.asarray(kwargs['up'], dtype=np.bool_)

        my = np.asarray(kwargs['mo_func'](x, params), dtype=float)
        if not np.isfinite(my).all():
            return np.inf, np.full(y.shape[0], np.inf)

        return _odr_core(my, x, y, x_err, y_err, w, up, float(params[0]), float(params[2]))

    @staticmethod
    def groth(**kwargs):
        """Groth exact-Poisson statistic; ``w`` and ``up`` are ignored."""

        x = np.asarray(kwargs['x'], dtype=float)
        y = np.asarray(kwargs['y'], dtype=float)

        my = np.asarray(kwargs['mo_func'](x, kwargs['params']), dtype=float)
        if not np.isfinite(my).all():
            return np.inf, np.full(y.shape[0], np.inf)

        return _groth_core(my, y)

    _allowed_stats = MappingProxyType(
        {
            'chi^2': chi_square.__func__,
            'chi^2f': chi_square_full.__func__,
            'logchi^2': log_chi_square.__func__,
            'vdr': vdr.__func__,
            'odr': odr.__func__,
            'groth': groth.__func__,
        }
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_statistic_residual.py -v`
Expected: all PASS (first run compiles numba kernels — a few seconds).

- [ ] **Step 5: Gate + commit**

```bash
cd /Users/junyang/Documents/python_works/curvefit
ruff check . && ruff format --check . && python -m pytest -q
git add curvefit/infer/statistic.py tests/test_statistic_residual.py
git commit -m "feat: statistics return (stat, residual) via njit cores + unified kwargs"
```
Note: existing `tests/test_statistic.py` (Task-2-era) calls the old positional signature `Statistic.chi_square(lin, params, ...)`. Update it to the keyword form `Statistic.chi_square(**_kwargs(...))[0]` (the loglike is `-0.5 * stat`), or fold its assertions into the new test and delete it. Do whichever keeps the suite green; commit the change.

---

### Task 2: `infer/pair.py` — `**kwargs` dispatch + `pseudo_residual`

**Files:**
- Replace: `curvefit/infer/pair.py`
- Test: `tests/test_pair.py` (existing; extend)

- [ ] **Step 1: Add the failing test (append to `tests/test_pair.py`)**

Append:
```python
def test_pair_pseudo_residual_matches_stat():
    import numpy as np

    from curvefit.data.data import Data, DataUnit
    from curvefit.infer.pair import Pair
    from curvefit.model.local import ln

    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    unit = DataUnit(x, y, yerr=np.full(4, 0.1), stat='chi^2')
    data = Data([('d', unit)])
    model = ln()
    model.params['k'].val = 2.5
    model.params['b'].val = 0.5
    pair = Pair(data, model)

    pr = pair.pseudo_residual
    # for chi^2, sum(pseudo_residual**2) equals the total stat
    assert np.isclose(np.sum(pr**2), pair.stat)
    assert np.isclose(pair.loglike, -0.5 * pair.stat)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_pair.py::test_pair_pseudo_residual_matches_stat -v`
Expected: FAIL — `Pair` has no `pseudo_residual`.

- [ ] **Step 3: Replace the file**

Overwrite `curvefit/infer/pair.py` with:
```python
"""Binding between a :class:`Data` and a :class:`Model` for fit evaluation.

Holds one model-data pair, evaluates each unit's statistic by dispatching
on its tag, and publishes the total statistic, per-bin pseudo-residuals
(for lmfit), sigma residuals (for plots), and the log-likelihood.
"""

import numpy as np

from ..data.data import Data
from ..model.model import Model
from .statistic import Statistic


class Pair:
    """Binds one ``Data`` and one ``Model`` and evaluates their joint statistic.

    Dispatches on each unit's ``stat`` tag via :attr:`_allowed_stats`; every
    statistic returns ``(stat, residual)``.

    Attributes:
        data: The bound :class:`~curvefit.data.data.Data`.
        model: The bound :class:`~curvefit.model.model.Model`.
    """

    _allowed_stats = Statistic._allowed_stats

    def __init__(self, data, model):
        """Pair ``data`` with ``model`` and wire the ``fit_with`` reference.

        Args:
            data: ``Data`` container.
            model: ``Model`` instance.
        """

        self._data = data
        self._model = model

        self._pair()

    @property
    def data(self):

        return self._data

    @data.setter
    def data(self, new_data):

        self._data = new_data

        self._pair()

    @property
    def model(self):

        return self._model

    @model.setter
    def model(self, new_model):

        self._model = new_model

        self._pair()

    def _pair(self):
        """Validate operand types and bind ``data.fit_with`` to the model.

        Raises:
            ValueError: If ``data``/``model`` are not the expected types.
        """

        if not isinstance(self.data, Data):
            raise ValueError('data argument should be Data type')

        if not isinstance(self.model, Model):
            raise ValueError('model argument should be Model type')

        self.data.fit_with = self.model

    def mo_func(self, x, params):
        """Set ``params`` on the model and evaluate it on x-grid ``x``."""

        self.model.at_par(params)

        return self.model.func(np.asarray(x, dtype=float)[:, None])

    @property
    def pvalues(self):
        """Full ordered raw-value vector (``Par.val``) of the model."""

        return [par.val for par in self.model.par.values()]

    def _kwargs(self, unit):
        """Assemble the shared statistic keyword bundle for one data unit."""

        return {
            'mo_func': self.mo_func,
            'params': self.pvalues,
            'x': unit.x,
            'y': unit.y,
            'x_err': unit.xerr,
            'y_err': unit.yerr,
            'w': unit.weight,
            'up': unit.up,
        }

    def _stat_calculate(self):

        return np.array(
            [self._allowed_stats[u.stat](**self._kwargs(u))[0] for u in self.data.data.values()],
            dtype=float,
        )

    def _pseudo_residual_calculate(self):

        return [self._allowed_stats[u.stat](**self._kwargs(u))[1] for u in self.data.data.values()]

    @property
    def stat_list(self):
        """Per-unit statistic values."""

        return self._stat_calculate()

    @property
    def pseudo_residual_list(self):
        """Per-unit signed pseudo-residual arrays."""

        return self._pseudo_residual_calculate()

    @property
    def weight_list(self):
        """Per-unit weights; point weights live inside ``w``, so this is ones."""

        return np.ones(len(self.data.data))

    @property
    def stat(self):
        """Total weighted statistic summed across units."""

        return np.sum(self.stat_list * self.weight_list)

    @property
    def pseudo_residual(self):
        """Pseudo-residual vector concatenated across units (lmfit target)."""

        return np.concatenate(self.pseudo_residual_list)

    @property
    def residual(self):
        """Per-unit sigma residuals ``(y - model) / yerr`` for diagnostics/plots."""

        out = list()
        for u in self.data.data.values():
            my = np.asarray(self.mo_func(u.x, self.pvalues), dtype=float)
            yerr = np.where(u.y < my, u.yerr[1], u.yerr[0])
            out.append((u.y - my) / yerr)

        return out

    @property
    def loglike_list(self):
        """Per-unit log-likelihood, derived as ``-0.5 * stat_list``."""

        return -0.5 * self.stat_list

    @property
    def loglike(self):
        """Total log-likelihood, derived as ``-0.5 * stat``."""

        return -0.5 * self.stat

    @property
    def npoint_list(self):
        """Per-unit number of fitted points."""

        return self.data.npoints

    @property
    def npoint(self):
        """Total number of fitted points across units."""

        return np.sum(self.npoint_list)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_pair.py -v`
Expected: all PASS (existing pair tests + the new one).

- [ ] **Step 5: Gate + commit**

```bash
cd /Users/junyang/Documents/python_works/curvefit
ruff check . && ruff format --check . && python -m pytest -q
git add curvefit/infer/pair.py tests/test_pair.py
git commit -m "feat: Pair exposes pseudo_residual and sigma residual"
```

---

### Task 3: `infer/infer.py` — `calc_*`, residual props, clean labels, hooks

**Files:**
- Modify: `curvefit/infer/infer.py`
- Test: `tests/test_infer.py` (existing; extend)

- [ ] **Step 1: Add the failing test (append to `tests/test_infer.py`)**

Append:
```python
def test_calc_stat_pseudo_residual_and_clean_labels():
    import numpy as np

    from curvefit.data.data import Data, DataUnit
    from curvefit.infer.infer import Infer
    from curvefit.model.local import ln

    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    unit = DataUnit(x, y, yerr=np.full(4, 0.1), stat='chi^2')
    data = Data([('d', unit)])
    model = ln()
    model.params['logv'].frozen = True
    infer = Infer([(data, model)])

    # at the truth, stat ~ 0 and pseudo_residual ~ 0
    assert np.isclose(infer.calc_stat([2.0, 1.0]), 0.0, atol=1e-6)
    assert np.allclose(infer.calc_pseudo_residual([2.0, 1.0]), 0.0, atol=1e-3)
    assert np.isclose(infer.calc_loglike([2.0, 1.0]), -0.5 * infer.calc_stat([2.0, 1.0]))
    # clean labels strip LaTeX markup
    assert all('$' not in pl and '\\' not in pl for pl in infer.clean_free_plabels)
    assert len(infer.clean_free_indexed_plabels) == infer.free_nparams
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_infer.py::test_calc_stat_pseudo_residual_and_clean_labels -v`
Expected: FAIL — `calc_stat`/`calc_pseudo_residual`/`clean_free_plabels` don't exist.

- [ ] **Step 3a: Add the override hooks to `Infer.__init__`**

In `curvefit/infer/infer.py`, the current `Infer.__init__` is:
```python
    def __init__(self, pairs=None):

        self.pairs = pairs
```
Replace it with:
```python
    def __init__(self, pairs=None):
        """Build an inference from ``(Data, Model)`` pairs.

        Args:
            pairs: ``None`` or a list of ``(Data, Model)`` / ``(Model, Data)``
                tuples.
        """

        self.loglike_func = None
        self.logprior_func = None

        self.pairs = pairs
```

- [ ] **Step 3b: Add clean-label properties**

In `curvefit/infer/infer.py`, immediately after the `free_plabels` property (which returns `self._free_plabels`), insert:
```python
    @property
    def clean_free_plabels(self):
        """:attr:`free_plabels` with LaTeX ``$``, ``{``, ``}`` and ``\\`` removed."""

        return [
            pl.replace('$', '').replace('{', '').replace('}', '').replace('\\', '')
            for pl in self._free_plabels
        ]

    @property
    def clean_free_indexed_plabels(self):
        """Clean free-parameter labels prefixed with their ``par#`` index."""

        return [
            f'p{key}({label})'
            for label, key in zip(self.clean_free_plabels, self.free_par.keys(), strict=False)
        ]
```

- [ ] **Step 3c: Add residual/pseudo_residual aggregate properties**

In `curvefit/infer/infer.py`, immediately after the existing `model_y` property, insert:
```python
    @property
    def residual(self):
        """Concatenated per-unit sigma residuals across all pairs."""

        return [rd for pair in self.Pair for rd in pair.residual]

    @property
    def pseudo_residual(self):
        """Concatenated per-unit pseudo-residual vector across all pairs."""

        return np.hstack([pair.pseudo_residual for pair in self.Pair])
```

- [ ] **Step 3d: Add the `calc_*` entry points**

In `curvefit/infer/infer.py`, immediately after the existing `_loglike` method, insert:
```python
    def calc_loglike(self, theta):
        """Apply ``theta`` and return the log-likelihood (or the user override)."""

        self.at_par(theta)

        if self.loglike_func is None:
            return self.loglike
        else:
            return self.loglike_func(self, theta)

    def calc_stat(self, theta):
        """Apply ``theta`` and return the summed fit statistic."""

        self.at_par(theta)

        return self.stat

    def calc_pseudo_residual(self, theta):
        """Apply ``theta`` and return the concatenated pseudo-residual vector."""

        self.at_par(theta)

        return self.pseudo_residual
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_infer.py -v`
Expected: all PASS.

- [ ] **Step 5: Gate + commit**

```bash
cd /Users/junyang/Documents/python_works/curvefit
ruff check . && ruff format --check . && python -m pytest -q
git add curvefit/infer/infer.py tests/test_infer.py
git commit -m "feat: Infer calc_stat/calc_loglike/calc_pseudo_residual + clean labels + hooks"
```

---

### Task 4: `infer/posterior.py` — `SampleAnalyzer` + `Posterior` + `Bootstrap`

Refactor so the shared analysis logic lives in a base class keyed by `sample_attribute`, then add `Bootstrap`. `emcee`/`multinest` keep returning `Posterior` unchanged.

**Files:**
- Replace: `curvefit/infer/posterior.py`
- Test: `tests/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bootstrap.py`:
```python
import numpy as np

from curvefit.data.data import Data, DataUnit
from curvefit.infer.infer import Infer
from curvefit.infer.posterior import Bootstrap, Posterior, SampleAnalyzer
from curvefit.model.local import ln


def _infer():
    x = np.linspace(0, 10, 20)
    y = 2.0 * x + 1.0
    unit = DataUnit(x, y, yerr=np.full(x.size, 0.3), stat='chi^2')
    data = Data([('d', unit)])
    model = ln()
    model.params['logv'].frozen = True
    return Infer([(data, model)])


def test_bootstrap_from_sample_matrix():
    infer = _infer()
    infer._you_free()
    nfree = infer.free_nparams
    # fabricate a bootstrap sample: first row is the best fit
    rng = np.random.default_rng(0)
    best = np.array([2.0, 1.0])
    draws = best + rng.normal(0, 0.05, size=(50, nfree))
    draws[0] = best
    loglike = np.array([infer.calc_loglike(t) for t in draws])
    infer.bootstrap_sample = np.hstack((draws, loglike[:, None]))

    boot = Bootstrap(infer)
    assert isinstance(boot, SampleAnalyzer)
    # best is the first row
    assert np.allclose(boot.par_best, [2.0, 1.0])
    assert len(boot.par_median) == nfree
    ci = boot.par_interval(0.6827)
    assert len(ci) == nfree


def test_posterior_still_works():
    infer = _infer()
    infer._you_free()
    nfree = infer.free_nparams
    rng = np.random.default_rng(1)
    draws = np.array([2.0, 1.0]) + rng.normal(0, 0.05, size=(200, nfree))
    loglike = np.array([infer.calc_loglike(t) for t in draws])
    infer.posterior_sample = np.hstack((draws, loglike[:, None]))

    post = Posterior(infer)
    assert len(post.par_best_ci) == nfree
    assert np.all(np.isfinite(post.par_mean))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_bootstrap.py -v`
Expected: FAIL — `SampleAnalyzer`/`Bootstrap` don't exist.

- [ ] **Step 3: Replace the file**

Overwrite `curvefit/infer/posterior.py` with:
```python
"""Posterior and bootstrap sample analysers over a fitted :class:`Infer`.

``SampleAnalyzer`` absorbs an ``Infer``, loads its ``(nsample, nfree+1)``
sample matrix (parameters plus a trailing log-probability column), attaches
a :class:`~curvefit.util.post.Post` to each free parameter, and exposes
point estimates, credible intervals, and information criteria. ``Posterior``
reads ``posterior_sample`` (nested sampling / MCMC); ``Bootstrap`` reads
``bootstrap_sample`` (maximum-likelihood resampling).
"""

from collections import OrderedDict

import numpy as np

from ..util.info import Info
from ..util.post import Post
from .infer import Infer


class SampleAnalyzer(Infer):
    """Shared analysis over a parameter sample absorbed from an ``Infer``.

    Subclasses set :attr:`sample_attribute` to the source ``Infer``
    attribute holding the sample matrix and :attr:`analyzer_type` to the
    display label.

    Attributes:
        sample_attribute: Name of the source attribute (set by subclass).
        analyzer_type: Human-readable label shown in ``__str__``.
    """

    sample_attribute = None
    analyzer_type = 'Sample Analysis Results'

    def __init__(self, infer):
        """Absorb ``infer`` and attach per-parameter posteriors.

        Args:
            infer: A fitted ``Infer`` carrying the sample matrix named by
                :attr:`sample_attribute`.
        """

        self.infer = infer

    @property
    def infer(self):

        return self._infer

    @infer.setter
    def infer(self, new_infer):
        """Validate, copy the infer state, load the sample, and analyse.

        Raises:
            TypeError: If ``new_infer`` is not an ``Infer``.
        """

        if not isinstance(new_infer, Infer):
            raise TypeError('expected an instance of Infer')

        self._infer = new_infer
        self.__dict__.update(new_infer.__dict__)

        self._load_sample()
        self._post()

    def _load_sample(self):
        """Load and validate the sample matrix from :attr:`sample_attribute`.

        Raises:
            AttributeError: If the attribute is undefined or absent.
            ValueError: If the matrix is not 2D with ``nfree + 1`` columns.
        """

        if self.sample_attribute is None:
            raise AttributeError('sample_attribute is not defined')

        self.sample = getattr(self, self.sample_attribute, None)
        if self.sample is None:
            raise AttributeError(f'{self.sample_attribute} is not available')

        self.sample = np.asarray(self.sample, dtype=float)
        if self.sample.ndim != 2:
            raise ValueError(f'{self.sample_attribute} is expected to be a 2D array')
        if self.sample.shape[1] != self.free_nparams + 1:
            raise ValueError(
                f'{self.sample_attribute} is expected to have {self.free_nparams + 1} columns'
            )

    def _post(self):
        """Attach a :class:`Post` to each free parameter and set the best fit."""

        for i in range(self.free_nparams):
            sample = self.sample[:, i].copy()
            logprob = self.sample[:, -1].copy()

            self.free_par[i + 1].post = Post(sample, logprob)

        self._set_best()

    def _set_best(self):
        """Set each free parameter's ``post.best_ci``; overridden per subclass."""

        raise NotImplementedError

    @property
    def par_mean(self):
        """Per-free-parameter posterior summaries.

        The ``par_mean``/``par_median``/``par_best``/``par_best_ci`` family
        each returns one value per free parameter, read from its ``Post``.
        """

        return [par.post.mean for par in self.free_par.values()]

    @property
    def par_median(self):

        return [par.post.median for par in self.free_par.values()]

    @property
    def par_best(self):

        return [par.post.best for par in self.free_par.values()]

    @property
    def par_best_ci(self):

        return [par.post.best_ci for par in self.free_par.values()]

    def par_quantile(self, q):
        """Per-parameter quantile at probability ``q``."""

        return [par.post.quantile(q) for par in self.free_par.values()]

    def par_interval(self, q):
        """Per-parameter central credible interval at probability ``q``."""

        return [par.post.interval(q) for par in self.free_par.values()]

    @property
    def par_Isigma(self):
        """Per-parameter 1/2/3-sigma central intervals."""

        return [par.post.Isigma for par in self.free_par.values()]

    @property
    def par_IIsigma(self):

        return [par.post.IIsigma for par in self.free_par.values()]

    @property
    def par_IIIsigma(self):

        return [par.post.IIIsigma for par in self.free_par.values()]

    def par_error(self, par, q=0.6827):
        """Return ``[low_gap, high_gap]`` per parameter around point estimate ``par``."""

        ci = self.par_interval(q)

        return [np.diff([c[0], p, c[1]]).tolist() for p, c in zip(par, ci, strict=False)]

    @property
    def max_loglike(self):
        """Log-likelihood at the best-fit parameter vector."""

        self.at_par(self.par_best)

        return self.loglike

    @property
    def aic(self):
        """Akaike information criterion at the best fit."""

        return -2 * self.max_loglike + 2 * self.free_nparams

    @property
    def aicc(self):
        """Small-sample corrected AIC."""

        return self.aic + 2 * self.free_nparams * (self.free_nparams + 1) / (
            self.npoint - self.free_nparams - 1
        )

    @property
    def bic(self):
        """Bayesian information criterion at the best fit."""

        return -2 * self.max_loglike + self.free_nparams * np.log(self.npoint)

    @property
    def lnZ(self):
        """Log-evidence if a sampler stored one, else ``None``."""

        try:
            return self.logevidence
        except AttributeError:
            return None

    @property
    def free_par_info(self):
        """Free-parameter summary table (mean/median/best/1-sigma CI)."""

        self._you_free()

        free_par_info = Info.list_dict_to_dict(self.free_params)

        del free_par_info['Posterior']
        del free_par_info['Mates']
        del free_par_info['Frozen']
        del free_par_info['Prior']
        del free_par_info['Value']

        free_par_info['Mean'] = [f'{par:.3f}' for par in self.par_mean]
        free_par_info['Median'] = [f'{par:.3f}' for par in self.par_median]
        free_par_info['Best'] = [f'{par:.3f}' for par in self.par_best_ci]
        free_par_info['1sigma CI'] = [
            '[{:.3f}, {:.3f}]'.format(*tuple(ci)) for ci in self.par_Isigma
        ]

        return Info.from_dict(free_par_info)

    @property
    def stat_info(self):
        """Goodness-of-fit table evaluated at the best fit."""

        self.at_par(self.par_best_ci)

        return Info.from_dict(self.all_stat)

    @property
    def IC_info(self):
        """Information-criteria table (AIC/AICc/BIC/lnZ)."""

        IC_info = OrderedDict()
        IC_info['AIC'] = [f'{self.aic:.2f}']
        IC_info['AICc'] = [f'{self.aicc:.2f}']
        IC_info['BIC'] = [f'{self.bic:.2f}']
        IC_info['lnZ'] = [f'{self.lnZ}' if self.lnZ is None else f'{self.lnZ:.2f}']

        return Info.from_dict(IC_info)

    def __str__(self):

        print(self.free_par_info.table)
        print(self.stat_info.table)
        print(self.IC_info.table)

        return ''


class Posterior(SampleAnalyzer):
    """Posterior analyser for nested-sampling / MCMC ``posterior_sample``."""

    sample_attribute = 'posterior_sample'
    analyzer_type = 'Posterior Results'

    def _set_best(self):
        """Pick the highest-logprob draw lying inside every 1-sigma interval."""

        argsort = np.argsort(self.sample[:, -1])[::-1]
        sort_sample = self.sample[:, 0:-1].copy()[argsort]

        for sample in sort_sample:
            if np.array(
                [(ci[0] <= sample[i] <= ci[1]) for i, ci in enumerate(self.par_interval(0.6827))]
            ).all():
                for par, value in zip(self.free_par.values(), sample, strict=False):
                    par.post.best_ci = value

                break

    @property
    def posterior_statistic(self):
        """Mean/median and 1/2/3-sigma quantiles of the full posterior matrix."""

        mean = np.mean(self.sample, axis=0)
        median = np.median(self.sample, axis=0)

        q = 68.27 / 100
        Isigma = np.quantile(self.sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        q = 95.45 / 100
        IIsigma = np.quantile(self.sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        q = 99.73 / 100
        IIIsigma = np.quantile(self.sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        return dict(
            [
                ('mean', mean),
                ('median', median),
                ('Isigma', Isigma),
                ('IIsigma', IIsigma),
                ('IIIsigma', IIIsigma),
            ]
        )


class Bootstrap(SampleAnalyzer):
    """Bootstrap analyser for maximum-likelihood ``bootstrap_sample``.

    The first row of the sample is the best-fit vector produced by
    :class:`~curvefit.infer.infer.MaxLikeFit`.
    """

    sample_attribute = 'bootstrap_sample'
    analyzer_type = 'Bootstrap Results'

    def _set_best(self):
        """Use the first sample row (the stored best fit) as ``best_ci``."""

        best = self.sample[0, 0:-1]

        for par, value in zip(self.free_par.values(), best, strict=False):
            par.post.best_ci = value
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_bootstrap.py -v`
Expected: both PASS.

- [ ] **Step 5: Gate + commit**

```bash
cd /Users/junyang/Documents/python_works/curvefit
ruff check . && ruff format --check . && python -m pytest -q
git add curvefit/infer/posterior.py tests/test_bootstrap.py
git commit -m "refactor: SampleAnalyzer base + Posterior + Bootstrap analysers"
```

---

### Task 5: `MaxLikeFit(Infer)` + export

**Files:**
- Modify: `curvefit/infer/infer.py` (append class)
- Modify: `curvefit/infer/__init__.py`
- Test: `tests/test_maxlikefit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_maxlikefit.py`:
```python
import numpy as np
import pytest

from curvefit.data.data import Data, DataUnit
from curvefit.infer.infer import MaxLikeFit
from curvefit.infer.posterior import Bootstrap
from curvefit.model.local import ln


def _data(seed=0, stat='chi^2'):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 40)
    yerr = np.full(x.size, 0.5)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, x.size)
    unit = DataUnit(x, y, yerr=yerr, stat=stat)
    return Data([('d', unit)])


def test_lmfit_recovers_linear():
    data = _data()
    model = ln()
    model.params['logv'].frozen = True
    fit = MaxLikeFit([(data, model)])
    boot = fit.lmfit()
    assert isinstance(boot, Bootstrap)
    assert boot.bootstrap_sample.shape[1] == fit.free_nparams + 1
    k, b = boot.par_best[0], boot.par_best[1]
    assert abs(k - 2.0) < 0.3
    assert abs(b - 1.0) < 1.0


def test_iminuit_recovers_linear():
    data = _data(seed=1)
    model = ln()
    model.params['logv'].frozen = True
    fit = MaxLikeFit([(data, model)])
    boot = fit.iminuit()
    assert isinstance(boot, Bootstrap)
    assert abs(boot.par_best[0] - 2.0) < 0.3


def test_iminuit_handles_chi2f_freevariance():
    data = _data(seed=2, stat='chi^2f')
    model = ln()  # k, b, logv all free
    fit = MaxLikeFit([(data, model)])
    boot = fit.iminuit()
    assert isinstance(boot, Bootstrap)
    assert abs(boot.par_best[0] - 2.0) < 0.5


def test_lmfit_rejects_free_variance_stats():
    data = _data(seed=3, stat='vdr')
    model = ln()
    fit = MaxLikeFit([(data, model)])
    with pytest.raises(ValueError):
        fit.lmfit()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_maxlikefit.py -v`
Expected: FAIL — `MaxLikeFit` not importable.

- [ ] **Step 3a: Append `MaxLikeFit` to `infer.py`**

Add to the end of `curvefit/infer/infer.py` (after `BayesInfer`):
```python
class MaxLikeFit(Infer):
    """:class:`Infer` extension for maximum-likelihood fits with bootstrap sampling.

    Provides :meth:`lmfit` (least-squares on the pseudo-residuals; only for
    pure chi-square statistics) and :meth:`iminuit` (scalar minimisation of
    the fit statistic; valid for every statistic). Both run the minimiser,
    build a covariance-driven bootstrap sample, and return a
    :class:`~curvefit.infer.posterior.Bootstrap`.
    """

    def __init__(self, pairs=None):
        """Initialise like :class:`Infer` and tag the inference type."""

        super().__init__(pairs=pairs)

        self.inference_type = 'Maximum Likelihood Estimation'

    def _make_bootstrap_sample(
        self, values, covar=None, errors=None, nsample=1000, random_seed=450001
    ):
        """Draw a covariance-respecting bootstrap sample and score each draw.

        Falls back to a diagonal covariance built from ``errors`` when
        ``covar`` is missing or non-finite. Draws outside any free
        parameter's range are rejected. The first row is the best fit.

        Args:
            values: Best-fit free-parameter vector.
            covar: Optional parameter covariance matrix.
            errors: Optional per-parameter uncertainties for the fallback.
            nsample: Target number of valid draws.
            random_seed: Seed for reproducibility.
        """

        values = np.asarray(values, dtype=float)
        ndim = values.size

        nsample = max(int(nsample), 1)

        if covar is not None:
            covar = np.asarray(covar, dtype=float)

        if covar is None or covar.shape != (ndim, ndim) or (not np.isfinite(covar).all()):
            msg = (
                'Covariance matrix is not provided or invalid. '
                'Using diagonal covariance with variances from errors or zeros.'
            )
            warnings.warn(msg, stacklevel=2)
            err = np.zeros(ndim, dtype=float) if errors is None else np.asarray(errors, dtype=float)
            err = np.where(np.isfinite(err), np.abs(err), 0.0)
            covar = np.diag(err * err)

        covar = 0.5 * (covar + covar.T)
        eigval, eigvec = np.linalg.eigh(covar)
        scale = np.max(np.abs(eigval)) if eigval.size else 1.0
        floor = np.finfo(float).eps * (scale if scale > 0 else 1.0)
        eigval = np.clip(eigval, floor, None)
        covar = eigvec @ np.diag(eigval) @ eigvec.T

        lower = np.array([pr[0] for pr in self.free_pranges], dtype=float)
        upper = np.array([pr[1] for pr in self.free_pranges], dtype=float)

        rng = np.random.default_rng(random_seed)

        param_sample = [values.copy()]
        tries = 0
        while len(param_sample) < nsample and tries < 10:
            batch_size = max(4 * (nsample - len(param_sample)), 128)
            draw = rng.multivariate_normal(values, covar, size=batch_size, check_valid='ignore')
            draw = np.atleast_2d(draw)

            inside = np.all((draw >= lower) & (draw <= upper), axis=1)
            param_sample.extend(draw[inside][: nsample - len(param_sample)])
            tries += 1

        if len(param_sample) < nsample:
            msg = f'Only {len(param_sample)} valid samples were generated after {tries} attempts.'
            warnings.warn(msg, stacklevel=2)
            param_sample = np.asarray(param_sample, dtype=float)
        else:
            param_sample = np.asarray(param_sample[:nsample], dtype=float)

        loglike_sample = np.array([self.calc_loglike(theta) for theta in param_sample], dtype=float)

        self.bootstrap_sample = np.hstack((param_sample, loglike_sample[:, None]))

        self.at_par(values)

    @staticmethod
    def _display_results(*objects):
        """Render each object with IPython when available, else ``print`` it."""

        valid_objects = [obj for obj in objects if obj is not None]

        try:
            from IPython.display import display
        except ImportError:
            for obj in valid_objects:
                print(obj)
            return

        for obj in valid_objects:
            display(obj)

    def _check_lmfit_safe(self):
        """Ensure every unit uses an lmfit-safe statistic.

        Raises:
            ValueError: If any unit's statistic is not in
                :data:`~curvefit.infer.statistic.LMFIT_SAFE_STATS`; use
                :meth:`iminuit` for those.
        """

        from .statistic import LMFIT_SAFE_STATS

        stats = [s for data in self.Data for s in data.stats]
        bad = sorted({s for s in stats if s not in LMFIT_SAFE_STATS})
        if bad:
            msg = (
                f'lmfit (least-squares) supports only {sorted(LMFIT_SAFE_STATS)}; '
                f'statistics {bad} fit a free variance -- use iminuit() instead.'
            )
            raise ValueError(msg)

    def lmfit_residual(self, params):
        """lmfit-facing residual callback; delegates to :meth:`calc_pseudo_residual`."""

        theta = [params[pl] for pl in self.clean_free_plabels]

        return self.calc_pseudo_residual(theta)

    def lmfit(self, savepath=None):
        """Run ``lmfit.minimize`` on the pseudo-residuals and bootstrap the result.

        Args:
            savepath: Optional directory for persisted bootstrap samples and
                summary JSON; ``None`` skips disk IO.

        Returns:
            A :class:`~curvefit.infer.posterior.Bootstrap`.

        Raises:
            ValueError: If any unit uses a non-lmfit-safe statistic.
        """

        import lmfit

        from .posterior import Bootstrap

        self._you_free()
        self._check_lmfit_safe()

        lmfit_params = lmfit.Parameters()

        for pl, pv, pr in zip(
            self.clean_free_plabels, self.free_pvalues, self.free_pranges, strict=False
        ):
            lmfit_params.add(pl, value=pv, min=pr[0], max=pr[1], vary=True)

        lmfit_result = lmfit.minimize(self.lmfit_residual, lmfit_params)

        self._display_results(lmfit_result)

        values = np.array([lmfit_result.params[pl].value for pl in self.clean_free_plabels])
        errors = np.array(
            [
                np.nan if lmfit_result.params[pl].stderr is None else lmfit_result.params[pl].stderr
                for pl in self.clean_free_plabels
            ]
        )
        covar = getattr(lmfit_result, 'covar', None)

        self._make_bootstrap_sample(values, covar=covar, errors=errors)

        maxlike_res = {'values': values, 'errors': errors, 'covar': covar}

        if savepath is not None:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            savepath_prefix = savepath + '/1-'

            np.savetxt(savepath_prefix + 'bootstrap_sample.txt', self.bootstrap_sample)
            with open(savepath_prefix + 'maxlike_res.json', 'w') as f:
                json.dump(maxlike_res, f, indent=4, cls=JsonEncoder)

        return Bootstrap(self)

    def iminuit_cost(self, *theta):
        """iminuit-facing cost; returns ``1e100`` when the statistic is non-finite."""

        cost = self.calc_stat(theta)

        if np.isfinite(cost):
            return float(cost)
        else:
            return 1e100

    def iminuit(self, savepath=None):
        """Run iminuit's ``migrad`` + ``hesse`` + ``minos`` and bootstrap the result.

        Args:
            savepath: Optional directory for persisted bootstrap samples and
                summary JSON.

        Returns:
            A :class:`~curvefit.infer.posterior.Bootstrap`.
        """

        import iminuit

        from .posterior import Bootstrap

        self._you_free()

        minuit = iminuit.Minuit(
            self.iminuit_cost, *self.free_pvalues, name=self.clean_free_indexed_plabels
        )
        minuit.errordef = 2 * iminuit.Minuit.LIKELIHOOD
        minuit.print_level = 0

        for pl, pr in zip(self.clean_free_indexed_plabels, self.free_pranges, strict=False):
            minuit.limits[pl] = pr

        minuit.migrad()
        minuit.hesse()
        minuit.minos()

        self._display_results(minuit)

        values = np.array([par.value for par in minuit.params])
        errors = np.array([par.error for par in minuit.params])
        minos_errors = np.array([par.merror for par in minuit.params])
        covar = None if minuit.covariance is None else np.asarray(minuit.covariance)

        self._make_bootstrap_sample(values, covar=covar, errors=errors)

        maxlike_res = {
            'values': values,
            'errors': errors,
            'minos_errors': minos_errors,
            'covar': covar,
        }

        if savepath is not None:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            savepath_prefix = savepath + '/1-'

            np.savetxt(savepath_prefix + 'bootstrap_sample.txt', self.bootstrap_sample)
            with open(savepath_prefix + 'maxlike_res.json', 'w') as f:
                json.dump(maxlike_res, f, indent=4, cls=JsonEncoder)

        return Bootstrap(self)
```

Note: `os`, `json`, `warnings`, `np`, `JsonEncoder` must already be imported at the top of `infer.py`. `os`, `json`, `np`, `JsonEncoder` are already imported. **Add `import warnings`** to the top import block if absent (check first; the multinest/emcee code may already import it — if not, add it grouped with the stdlib imports and re-run `ruff format`).

- [ ] **Step 3b: Export `MaxLikeFit`**

Replace `curvefit/infer/__init__.py` contents:
```python
from .pair import Pair
from .infer import Infer, BayesInfer, MaxLikeFit
from .statistic import Statistic
from .posterior import SampleAnalyzer, Posterior, Bootstrap
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_maxlikefit.py -v`
Expected: 4 PASS (lmfit + iminuit recover the line; iminuit handles chi^2f; lmfit rejects vdr).

- [ ] **Step 5: Gate + commit**

```bash
cd /Users/junyang/Documents/python_works/curvefit
ruff check . && ruff format --check . && python -m pytest -q
git add curvefit/infer/infer.py curvefit/infer/__init__.py tests/test_maxlikefit.py
git commit -m "feat: MaxLikeFit with lmfit/iminuit drivers + covariance bootstrap"
```

---

### Task 6: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Public imports + emcee regression**

Run:
```bash
cd /Users/junyang/Documents/python_works/curvefit
python -c "
import curvefit
from curvefit import MaxLikeFit, BayesInfer, Data, DataUnit
from curvefit.infer.posterior import Posterior, Bootstrap, SampleAnalyzer
print('imports ok')
"
python -m pytest tests/test_end_to_end.py -q
```
Expected: `imports ok`; the existing emcee end-to-end test still passes (it returns a `Posterior`).

- [ ] **Step 2: Full suite + ruff**

Run:
```bash
cd /Users/junyang/Documents/python_works/curvefit
ruff check . ; ruff format --check . ; python -m pytest -q
```
Expected: `All checks passed!`, all files formatted, all tests pass.

---

## Self-Review

**Spec coverage:** §2 statistic rewrite → Task 1 (njit cores, `(stat,residual)`, `**kwargs`, `LMFIT_SAFE_STATS`). §3 Pair → Task 2 (`pseudo_residual_*`, `stat_func` via `[0]`, `residual`). §4 Infer → Task 3 (`calc_*`, `clean_free_plabels`/`clean_free_indexed_plabels`, `pseudo_residual`/`residual`, `loglike_func`/`logprior_func`). §5 analyzer → Task 4 (`SampleAnalyzer`+`Posterior`+`Bootstrap`). §6 MaxLikeFit → Task 5 (`lmfit`/`iminuit`/`_make_bootstrap_sample`/`_display_results`, export). §7 validation → Tasks 1-6 tests + Task 6. Decisions 1-5 (keep normalization, lmfit-safe set, residual clip, njit, kwargs) all realized. ✓

**Placeholder scan:** No TBD/TODO; full code in every code step; the `additive.py`-style move is N/A here. The one conditional ("add `import warnings` if absent") names the exact action. ✓

**Type consistency:** `Statistic._allowed_stats` (Task 1) consumed by `Pair._allowed_stats` (Task 2). `Pair.pseudo_residual` consumed by `Infer.pseudo_residual` (Task 3) → `calc_pseudo_residual` → `MaxLikeFit.lmfit_residual` (Task 5). `Infer.clean_free_plabels`/`clean_free_indexed_plabels`/`calc_stat`/`calc_loglike` (Task 3) consumed by `MaxLikeFit` (Task 5). `bootstrap_sample` set by `_make_bootstrap_sample` (Task 5) read by `Bootstrap.sample_attribute` (Task 4). `LMFIT_SAFE_STATS` (Task 1) used by `_check_lmfit_safe` (Task 5). `_set_best` defined abstract in `SampleAnalyzer`, overridden in `Posterior`/`Bootstrap` (Task 4). ✓

**Cross-task dependency note:** Task 4 must precede Task 5 (MaxLikeFit imports `Bootstrap`). Tasks 1→2→3→4→5→6 is the required order.
