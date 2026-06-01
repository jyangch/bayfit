# curvefit Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the half-ported `curvefit` package into a working generic curve-fitting framework (data = `x, y, xerr, yerr`) that mirrors bayspec's structure and code style, reusing the existing generic `model`/`util` code and integrating the user's six-statistic `Stat` engine.

**Architecture:** Four blocks `util / data / model / infer` exactly like bayspec. `model/local.py`, `util/param|prior|post|info|tools|corner.py`, and `infer/posterior.py` are reused as-is. `data/data.py` is rewritten for point data (`x, y, xerr, yerr, weight, up, stat`); `infer/statistic.py` becomes the user's six statistics wrapped in bayspec-style dict dispatch; `infer/pair.py` bridges `Model.func(X)` to the statistic signature `func(mo_func, params, x, y, x_err, y_err, w, up)`; `infer/infer.py` swaps spectral properties for `x/y/model_y`; `util/plot.py` is rewritten as a basic curve plotter.

**Tech Stack:** Python 3.12, numpy 2.2, scipy, pandas, emcee, matplotlib, plotly, pytest.

**Key conventions (do not violate):**
- NEVER modify the sibling `bayspec` package. Only edit under `/Users/junyang/Documents/python_works/curvefit`.
- Match bayspec code style: `OrderedDict` param containers, property/setter pattern, `Info` tables, blank-line spacing, no type hints.
- Model input is a 2-D array `X` with `X[:, 0]` the x-axis (already established in `model/local.py`).
- Statistic signature is fixed: `func(mo_func, params, x, y, x_err, y_err, w, up)` returning a **log-likelihood**. `x_err`/`y_err` are `[low, high]`; `up` is a per-point upper-limit boolean.
- The model↔statistic bridge: `mo_func(x, params)` sets `model` params via `model.at_par(params)` then returns `model.func(x[:, None])`. `params` = the model's full ordered parameter vector; `logv` (extra variance) is the model's **last** parameter.

---

### Task 0: Project setup — git, test scaffold

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`

- [ ] **Step 1: Initialize git and ignore caches**

Run:
```bash
cd /Users/junyang/Documents/python_works/curvefit
git init
git add -A
git commit -m "chore: snapshot half-ported curvefit before framework work"
```
Expected: a repository is created and the initial commit succeeds. (`.gitignore` already exists.)

- [ ] **Step 2: Create the test package and pytest config**

Create `tests/__init__.py` (empty file).

Create `pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

Create `tests/conftest.py`:
```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 3: Verify pytest collects nothing yet (no errors)**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest`
Expected: "no tests ran" (exit code 5 is fine), no collection errors.

- [ ] **Step 4: Commit**

```bash
git add tests pytest.ini
git commit -m "test: add pytest scaffold"
```

---

### Task 1: Fix `CompositeModel.func` signature in `model/model.py`

`CompositeModel.func` still uses the spectral `(E, T, O)` signature; the base class and all `local.py` models use `func(self, X)`. This makes any composed model (`m1 + m2`, `m1 * m2`) crash.

**Files:**
- Modify: `curvefit/model/model.py:337-348`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model.py`:
```python
import numpy as np
from curvefit.model.local import ln, pl


def test_single_model_eval():
    m = ln()
    m.params['k'].value = 2.0
    m.params['b'].value = 1.0
    X = np.array([0.0, 1.0, 2.0])[:, None]
    assert np.allclose(m.func(X), [1.0, 3.0, 5.0])


def test_composite_add_eval():
    a = ln()
    a.params['k'].value = 1.0
    a.params['b'].value = 0.0
    b = pl()
    b.params['alpha'].value = 1.0
    b.params['logA'].value = 0.0   # amplitude 1
    model = a + b
    X = np.array([1.0, 2.0, 3.0])[:, None]
    # ln = x ; pl = x ; sum = 2x
    assert np.allclose(model.func(X), [2.0, 4.0, 6.0])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_model.py -v`
Expected: `test_composite_add_eval` FAILS (TypeError: func() missing/extra args, or import error from the broken package). `test_single_model_eval` may also error if the package import chain is still broken — that is fixed in Task 4; run this file again after Task 4 if so.

- [ ] **Step 3: Apply the fix**

In `curvefit/model/model.py`, replace the `CompositeModel.func` method (currently lines 337-348):
```python
    def func(self, E, T=None, O=None):
        
        if self.op == '+':
            return self.m1.func(E, T, O) + self.m2.func(E, T, O)
        elif self.op == '-':
            return self.m1.func(E, T, O) - self.m2.func(E, T, O)
        elif self.op == '*':
            return self.m1.func(E, T, O) * self.m2.func(E, T, O)
        elif self.op == '/':
            return self.m1.func(E, T, O) / self.m2.func(E, T, O)
        else:
            raise ValueError(f'Unknown operation: {self.op}')
```
with:
```python
    def func(self, X):
        
        if self.op == '+':
            return self.m1.func(X) + self.m2.func(X)
        elif self.op == '-':
            return self.m1.func(X) - self.m2.func(X)
        elif self.op == '*':
            return self.m1.func(X) * self.m2.func(X)
        elif self.op == '/':
            return self.m1.func(X) / self.m2.func(X)
        else:
            raise ValueError(f'Unknown operation: {self.op}')
```

- [ ] **Step 4: Run the test (will pass after Task 4 fixes the import chain)**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_model.py -v`
Expected: PASS if the package already imports; otherwise these pass after Task 4. Do not block — proceed.

- [ ] **Step 5: Commit**

```bash
git add curvefit/model/model.py tests/test_model.py
git commit -m "fix: CompositeModel.func uses generic X signature"
```

---

### Task 2: Rewrite `infer/statistic.py` with the six-statistic `Statistic` engine

Replace the spectral `Statistic` class with the user's six curve-fit statistics (formulas verbatim), wrapped bayspec-style: a class of `@staticmethod`s, dispatched by name from `Pair`. Fix `np.math.factorial` → `math.factorial` (numpy 2.x removed `np.math`).

**Files:**
- Replace: `curvefit/infer/statistic.py` (entire file)
- Test: `tests/test_statistic.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_statistic.py`:
```python
import numpy as np
from curvefit.infer.statistic import Statistic


def lin(x, params):
    return params[0] * np.asarray(x) + params[1]


def test_chi_square_matches_manual():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.1, 2.1, 3.9])
    yerr = np.vstack([np.full(3, 0.1), np.full(3, 0.1)])
    xerr = np.zeros((2, 3))
    w = np.ones(3)
    up = np.zeros(3, dtype=bool)
    params = [2.0, 0.0]
    my = lin(x, params)
    expected = -0.5 * np.sum((y - my) ** 2 / 0.1 ** 2)
    got = Statistic.chi_square(lin, params, x, y, xerr, yerr, w, up)
    assert np.isclose(got, expected)


def test_upper_limit_below_model_is_allowed():
    # an upper-limit point with y >= model contributes 0 (consistent)
    x = np.array([1.0])
    y = np.array([5.0])
    yerr = np.vstack([[0.1], [0.1]])
    xerr = np.zeros((2, 1))
    w = np.ones(1)
    up = np.array([True])
    params = [1.0, 0.0]   # model = 1.0 < y=5.0  -> allowed -> S=0
    got = Statistic.chi_square(lin, params, x, y, xerr, yerr, w, up)
    assert got == 0.0


def test_upper_limit_above_model_is_forbidden():
    x = np.array([1.0])
    y = np.array([0.5])
    yerr = np.vstack([[0.1], [0.1]])
    xerr = np.zeros((2, 1))
    w = np.ones(1)
    up = np.array([True])
    params = [1.0, 0.0]   # model = 1.0 > y=0.5 -> forbidden -> -inf
    got = Statistic.chi_square(lin, params, x, y, xerr, yerr, w, up)
    assert got == -np.inf


def test_all_six_statistics_callable_finite():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.1, 2.0, 3.2])
    yerr = np.vstack([np.full(3, 0.2), np.full(3, 0.2)])
    xerr = np.vstack([np.full(3, 0.1), np.full(3, 0.1)])
    w = np.ones(3)
    up = np.zeros(3, dtype=bool)
    # vdr/odr expect [k, b, logv]; chi^2f expects logv as last
    params = [1.0, 0.0, -1.0]
    for name in ['chi^2', 'chi^2f', 'logchi^2', 'vdr', 'odr', 'groth']:
        func = Statistic._allowed_stats[name]
        val = func(lin, params, x, y, xerr, yerr, w, up)
        assert np.isfinite(val), name
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_statistic.py -v`
Expected: FAIL — current `Statistic` has no `chi_square`/`_allowed_stats` (it has `Gstat`, etc.).

- [ ] **Step 3: Replace the file**

Overwrite `curvefit/infer/statistic.py` entirely with:
```python
import numpy as np
from math import factorial
from decimal import Decimal



class Statistic(object):
    
    @staticmethod
    def chi_square(mo_func, params, x, y, x_err, y_err, w, up):
        
        x, y = np.array(x), np.array(y)
        x_err, y_err = np.array(x_err), np.array(y_err)
        my = mo_func(x, params)

        yerr = (y < my).astype(int) * y_err[1] + (y >= my).astype(int) * y_err[0]
        S = w * (y - my) ** 2 / yerr ** 2
        S[(y >= my) & up] = 0
        S[(y < my) & up] = np.inf
        return -0.5 * np.sum(S)


    @staticmethod
    def chi_square_full(mo_func, params, x, y, x_err, y_err, w, up):
        
        x, y = np.array(x), np.array(y)
        x_err, y_err = np.array(x_err), np.array(y_err)
        my = mo_func(x, params[:-1])
        logv = params[-1]

        yerr = (y < my).astype(int) * y_err[1] + (y >= my).astype(int) * y_err[0]
        sigma2 = yerr ** 2 + 10 ** (2 * logv)
        S = w * ((y - my) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
        S[(y >= my) & up] = 0
        S[(y < my) & up] = np.inf
        return -0.5 * np.sum(S)


    @staticmethod
    def log_chi_square(mo_func, params, x, y, x_err, y_err, w, up):
        
        x, y = np.array(x), np.array(y)
        x_err, y_err = np.array(x_err), np.array(y_err)
        logx, logy = np.log10(x), np.log10(y)
        logy_err = [logy - np.log10(y - y_err[0]), np.log10(y + y_err[1]) - logy]
        my = mo_func(x, params)
        logmy = np.log10(my)

        logyerr = np.array(logy < logmy).astype(int) * logy_err[1] \
            + np.array(logy >= logmy).astype(int) * logy_err[0]
        S = w * (logy - logmy) ** 2 / logyerr ** 2
        S[(logy >= logmy) & up] = 0
        S[(logy < logmy) & up] = np.inf
        return -0.5 * np.sum(S)


    @staticmethod
    def vdr(mo_func, params, x, y, x_err, y_err, w, up):
        
        x, y = np.array(x), np.array(y)
        x_err, y_err = np.array(x_err), np.array(y_err)
        my = mo_func(x, params)

        k, b, log_v = params[0], params[1], params[2]
        if k > 0:
            xerr = (y < my).astype(int) * x_err[0] + (y >= my).astype(int) * x_err[1]
        else:
            xerr = (y < my).astype(int) * x_err[1] + (y >= my).astype(int) * x_err[0]
        yerr = (y < my).astype(int) * y_err[1] + (y >= my).astype(int) * y_err[0]
        sigma2 = np.exp(2 * log_v) + k ** 2 * xerr ** 2 + yerr ** 2
        S = w * ((y - my) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
        S[(y >= my) & up] = 0
        S[(y < my) & up] = np.inf
        return -0.5 * np.sum(S)


    @staticmethod
    def odr(mo_func, params, x, y, x_err, y_err, w, up):
        
        x, y = np.array(x), np.array(y)
        x_err, y_err = np.array(x_err), np.array(y_err)
        my = mo_func(x, params)

        k, b, log_v = params[0], params[1], params[2]
        if k > 0:
            xerr = (y < my).astype(int) * x_err[0] + (y >= my).astype(int) * x_err[1]
        else:
            xerr = (y < my).astype(int) * x_err[1] + (y >= my).astype(int) * x_err[0]
        yerr = (y < my).astype(int) * y_err[1] + (y >= my).astype(int) * y_err[0]

        delta2 = (y - my) ** 2 / (k ** 2 + 1)
        sigma2 = (k ** 2 * xerr ** 2 + yerr ** 2) / (k ** 2 + 1) + np.exp(2 * log_v)
        S = w * (delta2 / sigma2 + np.log(2 * np.pi * sigma2))
        S[(y >= my) & up] = 0
        S[(y < my) & up] = np.inf
        return -0.5 * np.sum(S)


    @staticmethod
    def groth(mo_func, params, x, y, x_err, y_err, w, up):
        
        x, y = np.array(x), np.array(y)
        x_err, y_err = np.array(x_err), np.array(y_err)
        my = mo_func(x, params)

        lnL = 0
        for i in range(len(y)):
            yi = Decimal(float(y[i]))
            myi = Decimal(float(my[i]))

            m = 0
            dft = 0
            while True:
                dfti = (yi ** m) * (myi ** m) / (factorial(m)) ** 2
                if dfti > 1e-20:
                    dft += dfti
                    m += 1
                else:
                    break
            Li = np.exp(-(float(yi) + float(myi))) * float(dft)
            lnL += np.log(Li)
        return lnL


    _allowed_stats = {'chi^2': chi_square.__func__, 
                      'chi^2f': chi_square_full.__func__, 
                      'logchi^2': log_chi_square.__func__, 
                      'vdr': vdr.__func__, 
                      'odr': odr.__func__, 
                      'groth': groth.__func__}
```

Note: `chi_square.__func__` unwraps the `staticmethod` object so the dict stores a plain callable (referencing names inside the class body, before the class object exists).

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_statistic.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add curvefit/infer/statistic.py tests/test_statistic.py
git commit -m "feat: six curve-fit statistics with dict dispatch"
```

---

### Task 3: Rewrite `data/data.py` for `x, y, xerr, yerr` point data

`DataUnit` holds one dataset; `Data` is an ordered collection. Errors normalize to `(2, npoint)` `[low, high]`; `weight`→per-point array; `up`→per-point bool. Loaders: direct, dict, DataFrame, json, csv.

**Files:**
- Replace: `curvefit/data/data.py` (entire file)
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_data.py`:
```python
import json
import numpy as np
import pandas as pd
from curvefit.data.data import DataUnit, Data


def test_defaults_and_shapes():
    u = DataUnit([1, 2, 3], [10, 20, 30])
    assert u.npoint == 3
    assert u.xerr.shape == (2, 3)
    assert u.yerr.shape == (2, 3)
    assert np.all(u.xerr == 0)
    assert np.all(u.yerr == 1)        # no yerr -> OLS
    assert np.all(u.up == False)
    assert np.all(u.weight == 1)
    assert u.stat == 'chi^2'


def test_symmetric_and_asymmetric_errors():
    u = DataUnit([1, 2], [3, 4], yerr=[0.5, 0.5])
    assert np.allclose(u.yerr, [[0.5, 0.5], [0.5, 0.5]])
    a = DataUnit([1, 2], [3, 4], yerr=[[0.1, 0.2], [0.3, 0.4]])
    assert np.allclose(a.yerr, [[0.1, 0.2], [0.3, 0.4]])   # (2, n) low/high


def test_from_dict():
    u = DataUnit.from_dict({'x': [1, 2], 'y': [3, 4], 'yerr': [0.1, 0.2], 'stat': 'chi^2f'})
    assert u.stat == 'chi^2f'
    assert np.allclose(u.y, [3, 4])


def test_from_dataframe_asymmetric():
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4],
                       'yl': [0.1, 0.2], 'yh': [0.3, 0.4]})
    u = DataUnit.from_dataframe(df, yerr_low='yl', yerr_high='yh')
    assert np.allclose(u.yerr, [[0.1, 0.2], [0.3, 0.4]])


def test_from_csv(tmp_path):
    p = tmp_path / 'd.csv'
    pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6], 'yerr': [0.1, 0.1, 0.1]}).to_csv(p, index=False)
    u = DataUnit.from_csv(str(p))
    assert u.npoint == 3
    assert np.allclose(u.yerr[0], 0.1)


def test_from_json(tmp_path):
    p = tmp_path / 'd.json'
    json.dump({'x': [1, 2], 'y': [3, 4], 'yerr': [0.1, 0.1]}, open(p, 'w'))
    u = DataUnit.from_json(str(p))
    assert u.npoint == 2


def test_data_container():
    u1 = DataUnit([1, 2], [3, 4])
    u2 = DataUnit([1, 2], [3, 4], stat='vdr', up=[False, True])
    data = Data([('a', u1), ('b', u2)])
    assert data.exprs == ['a', 'b']
    assert data.stats == ['chi^2', 'vdr']
    assert list(data.npoints) == [2, 2]
    assert 'a' in data
    assert data['b'].stat == 'vdr'
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_data.py -v`
Expected: FAIL — current `DataUnit` signature is `(src, bkg, ...)`, no `from_dict`/`from_csv`.

- [ ] **Step 3: Replace the file**

Overwrite `curvefit/data/data.py` entirely with:
```python
import json
import inspect
import numpy as np
import pandas as pd
from ..util.info import Info
from collections import OrderedDict



class Data(object):
    
    def __init__(self, data=None):
        
        self.data = data
        

    @property
    def data(self):
        
        return self._data
    
    
    @data.setter
    def data(self, new_data):
        
        self._data = OrderedDict()

        if new_data is None:
            pass
            
        elif isinstance(new_data, list):
            for item in new_data:
                if isinstance(item, tuple):
                    self._setitem(*item)
                    
            self._extract()
                    
        elif isinstance(new_data, dict):
            for item in new_data.items():
                self._setitem(*item)
                
            self._extract()
            
        else:
            raise ValueError('unsupported data type')

        
    def _setitem(self, key, value):
        
        if not isinstance(value, DataUnit):
            raise ValueError('value parameter should be DataUnit type')
        
        value.name = key
        self._data[key] = value


    def _extract(self):
        
        if self.data is None:
            raise ValueError('data is None')
        
        self.exprs = [key for key in self.data.keys()]
        self.stats = [unit.stat for unit in self.data.values()]
        self.npoints = np.array([unit.npoint for unit in self.data.values()])


    @property
    def expr(self):
        
        return self.get_obj_name() or 'data'


    @property
    def pdicts(self):
        
        return OrderedDict()


    @property
    def info(self):
        
        info_dict = OrderedDict()
        info_dict['Name'] = [key for key in self.data.keys()]
        info_dict['Npoint'] = [unit.npoint for unit in self.data.values()]
        info_dict['Statistic'] = [unit.stat for unit in self.data.values()]
        info_dict['Upperlimit'] = [int(np.sum(unit.up)) for unit in self.data.values()]

        return Info.from_dict(info_dict)


    @property
    def fit_with(self):
        
        try:
            return self._fit_with
        except AttributeError:
            raise AttributeError('no model fit with')
    
    
    @fit_with.setter
    def fit_with(self, new_model):
        
        from ..model.model import Model
        
        self._fit_with = new_model
        
        if not isinstance(self._fit_with, Model): 
            raise ValueError('fit_with argument should be Model type!')
        
        try:
            self._fit_with.fit_to
        except AttributeError:
            self._fit_with.fit_to = self
        else:
            if self._fit_with.fit_to != self:
                self._fit_with.fit_to = self


    def get_obj_name(self):
        
        frame = inspect.currentframe()
        
        possible_var_names = []
        
        while frame:
            local_vars = frame.f_locals.items()
            var_names = [var_name for var_name, var_val in local_vars if var_val is self]
            if var_names:
                possible_var_names.extend(var_names)
            frame = frame.f_back
        
        if possible_var_names:
            return possible_var_names[-1]
        
        return None


    def __getitem__(self, key):
        
        return self._data[key]


    def __setitem__(self, key, value):
        
        self._setitem(key, value)
        self._extract()


    def __delitem__(self, key):
        
        del self._data[key]
        self._extract()


    def __contains__(self, key):
        
        return key in self._data


    def __str__(self):
        
        print(self.info.table)
        
        return ''



class DataUnit(object):
    
    def __init__(
        self, 
        x, 
        y, 
        xerr=None, 
        yerr=None, 
        weight=1, 
        up=None, 
        stat='chi^2', 
        name=None
        ):
        
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.npoint = self.x.shape[0]
        
        self.xerr = self._normalize_err(xerr, default=0.0)
        self.yerr = self._normalize_err(yerr, default=1.0)
        self.weight = self._normalize_weight(weight)
        self.up = self._normalize_up(up)
        self.stat = stat
        
        if name is not None:
            self.name = name


    def _normalize_err(self, err, default):
        
        n = self.npoint
        
        if err is None:
            arr = np.full(n, default, dtype=float)
            return np.vstack([arr, arr])
        
        err = np.asarray(err, dtype=float)
        
        if err.ndim == 0:
            arr = np.full(n, float(err))
            return np.vstack([arr, arr])
        
        if err.ndim == 1:
            if err.shape[0] != n:
                raise ValueError('error length does not match data')
            return np.vstack([err, err])
        
        if err.ndim == 2:
            if err.shape == (2, n):
                return err.astype(float)
            if err.shape == (n, 2):
                return err.T.astype(float)
            raise ValueError('2D error should have shape (2, npoint) or (npoint, 2)')
        
        raise ValueError('unsupported error shape')


    def _normalize_weight(self, weight):
        
        n = self.npoint
        
        if np.isscalar(weight):
            return np.full(n, float(weight))
        
        weight = np.asarray(weight, dtype=float)
        if weight.shape[0] != n:
            raise ValueError('weight length does not match data')
        
        return weight


    def _normalize_up(self, up):
        
        n = self.npoint
        
        if up is None:
            return np.zeros(n, dtype=bool)
        
        up = np.asarray(up, dtype=bool)
        if up.shape[0] != n:
            raise ValueError('up length does not match data')
        
        return up


    @classmethod
    def from_dict(cls, d, **kwargs):
        
        return cls(
            d['x'], 
            d['y'], 
            xerr=d.get('xerr'), 
            yerr=d.get('yerr'), 
            weight=d.get('weight', 1), 
            up=d.get('up'), 
            stat=d.get('stat', 'chi^2'), 
            name=d.get('name'), 
            **kwargs)


    @classmethod
    def from_dataframe(
        cls, 
        df, 
        x='x', 
        y='y', 
        xerr=None, 
        yerr=None, 
        xerr_low=None, 
        xerr_high=None, 
        yerr_low=None, 
        yerr_high=None, 
        weight=None, 
        up=None, 
        stat='chi^2', 
        name=None
        ):
        
        def col(c):
            return None if c is None else np.asarray(df[c], dtype=float)
        
        xe = cls._cols_to_err(col(xerr), col(xerr_low), col(xerr_high))
        ye = cls._cols_to_err(col(yerr), col(yerr_low), col(yerr_high))
        w = 1 if weight is None else np.asarray(df[weight], dtype=float)
        u = None if up is None else np.asarray(df[up], dtype=bool)
        
        return cls(col(x), col(y), xerr=xe, yerr=ye, weight=w, up=u, stat=stat, name=name)


    @staticmethod
    def _cols_to_err(sym, low, high):
        
        if low is not None and high is not None:
            return np.vstack([low, high])
        if sym is not None:
            return sym
        return None


    @classmethod
    def from_json(cls, path, **kwargs):
        
        with open(path) as f:
            d = json.load(f)
        
        return cls.from_dict(d, **kwargs)


    @classmethod
    def from_csv(cls, path, **kwargs):
        
        df = pd.read_csv(path)
        
        return cls.from_dataframe(df, **kwargs)


    @property
    def name(self):
        
        try:
            return self._name
        except AttributeError:
            return self.get_obj_name()
    
    
    @name.setter
    def name(self, new_name):
        
        self._name = new_name


    @property
    def info(self):
        
        info_dict = OrderedDict()
        info_dict['npoint'] = self.npoint
        info_dict['stat'] = self.stat
        info_dict['upperlimit'] = int(np.sum(self.up))
        
        info_dict = OrderedDict([('property', list(info_dict.keys())), 
                                 (self.name, list(info_dict.values()))])
        
        return Info.from_dict(info_dict)


    def get_obj_name(self):
        
        frame = inspect.currentframe()
        
        possible_var_names = []
        
        while frame:
            local_vars = frame.f_locals.items()
            var_names = [var_name for var_name, var_val in local_vars if var_val is self]
            if var_names:
                possible_var_names.extend(var_names)
            frame = frame.f_back
        
        if possible_var_names:
            return possible_var_names[-1]
        
        return None


    def __str__(self):
        
        print(self.info.table)
        
        return ''
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_data.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add curvefit/data/data.py tests/test_data.py
git commit -m "feat: generic x/y/xerr/yerr Data and DataUnit with loaders"
```

---

### Task 4: Fix `data/__init__.py` and verify the package imports

`data/__init__.py` imports `spectrum`/`response` modules that do not exist, breaking `import curvefit`. Remove those imports. `plot.py` also imports them — temporarily this is fixed in Task 8; to import the package now we also stub `Plot` consumption order. We fix `plot.py` properly in Task 8, but `data/__init__.py` must change now so data tests run.

**Files:**
- Replace: `curvefit/data/__init__.py`
- Test: `tests/test_import_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_import_data.py`:
```python
def test_import_data_module():
    from curvefit.data.data import Data, DataUnit
    from curvefit.data import DataUnit as DU
    assert DU is DataUnit
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_import_data.py -v`
Expected: FAIL — `from curvefit.data import ...` triggers `data/__init__.py` which imports the missing `spectrum`.

- [ ] **Step 3: Replace the file**

Overwrite `curvefit/data/__init__.py` with:
```python
from .data import DataUnit, Data
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_import_data.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add curvefit/data/__init__.py tests/test_import_data.py
git commit -m "fix: data/__init__ no longer imports missing spectrum/response"
```

---

### Task 5: Rewrite `infer/pair.py` — model↔statistic bridge

`Pair` binds one `Data` + one `Model`, builds `mo_func(x, params)`, and computes per-unit log-likelihood by dispatching each unit's `stat` name to `Statistic`. `stat_list` is `-2 * loglike` per unit (so the existing `stat/dof` reporting in `infer.py` keeps working). Per-unit weight is 1 (point weights live inside the statistic's `w`).

**Files:**
- Replace: `curvefit/infer/pair.py` (entire file)
- Test: `tests/test_pair.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pair.py`:
```python
import numpy as np
from curvefit.data.data import Data, DataUnit
from curvefit.model.local import ln
from curvefit.infer.pair import Pair


def test_loglike_on_exact_linear_data():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    yerr = np.full(4, 0.1)
    unit = DataUnit(x, y, yerr=yerr, stat='chi^2')
    data = Data([('d', unit)])

    model = ln()
    model.params['k'].value = 2.0
    model.params['b'].value = 1.0
    model.params['logv'].frozen = True   # unused by chi^2

    pair = Pair(data, model)
    # model matches data exactly -> chi^2 ~ 0 -> loglike ~ 0
    assert np.isclose(pair.loglike, 0.0)
    assert np.isclose(pair.stat, 0.0)
    assert pair.npoint == 4


def test_loglike_penalizes_bad_fit():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    unit = DataUnit(x, y, yerr=np.full(4, 0.1), stat='chi^2')
    data = Data([('d', unit)])

    model = ln()
    model.params['k'].value = 0.0
    model.params['b'].value = 0.0
    pair = Pair(data, model)
    assert pair.loglike < 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_pair.py -v`
Expected: FAIL — current `Pair` calls `self.model.integ`, `data.src_counts`, etc.

- [ ] **Step 3: Replace the file**

Overwrite `curvefit/infer/pair.py` entirely with:
```python
import numpy as np
from ..data.data import Data
from ..model.model import Model
from .statistic import Statistic



class Pair(object):
    
    _allowed_stats = Statistic._allowed_stats

    def __init__(self, data, model):
        
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
        
        if not isinstance(self.data, Data):
            raise ValueError('data argument should be Data type')
        
        if not isinstance(self.model, Model):
            raise ValueError('model argument should be Model type')
        
        self.data.fit_with = self.model


    def mo_func(self, x, params):
        
        self.model.at_par(params)
        
        return self.model.func(np.asarray(x, dtype=float)[:, None])


    @property
    def pvalues(self):
        
        return [par.value for par in self.model.par.values()]


    def _stat_calculate(self):
        
        params = self.pvalues
        
        loglike = list()
        for unit in self.data.data.values():
            func = self._allowed_stats[unit.stat]
            loglike.append(func(self.mo_func, params, 
                                unit.x, unit.y, unit.xerr, unit.yerr, 
                                unit.weight, unit.up))
            
        return np.array(loglike, dtype=float)


    @property
    def loglike_list(self):
        
        return self._stat_calculate()
    
    
    @property
    def loglike(self):
        
        return np.sum(self.loglike_list)
    
    
    @property
    def stat_list(self):
        
        return -2 * self.loglike_list
    
    
    @property
    def stat(self):
        
        return np.sum(self.stat_list * self.weight_list)
    
    
    @property
    def weight_list(self):
        
        return np.ones(len(self.data.data))
    
    
    @property
    def npoint_list(self):
        
        return self.data.npoints
    
    
    @property
    def npoint(self):
        
        return np.sum(self.npoint_list)
```

Note: `_allowed_stats` is bound to `Statistic._allowed_stats` (the dict of plain callables created in Task 2).

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_pair.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add curvefit/infer/pair.py tests/test_pair.py
git commit -m "feat: Pair bridges Model.func to statistic engine"
```

---

### Task 6: Adapt `infer/infer.py` — generic data/model properties

`Infer` keeps `emcee`/`multinest`/`minimize`/`calc-loglike` machinery intact. Only the spectral data/model properties (`data_chbin_mean` … `model_cts_to_pht`, lines 294-399) are replaced with generic `data_x / data_y / data_xerr / data_yerr / data_up / model_y`. `_extract` already reads `data.exprs`, and `all_stat` reads `dt.exprs`/`dt.stats` — all provided by the new `Data`.

**Files:**
- Modify: `curvefit/infer/infer.py:294-399`
- Test: `tests/test_infer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_infer.py`:
```python
import numpy as np
from curvefit.data.data import Data, DataUnit
from curvefit.model.local import ln
from curvefit.infer.infer import Infer


def _make_infer():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    unit = DataUnit(x, y, yerr=np.full(4, 0.1), stat='chi^2')
    data = Data([('d', unit)])
    model = ln()
    model.params['logv'].frozen = True
    return Infer([(data, model)]), data, model


def test_free_params_and_dof():
    infer, data, model = _make_infer()
    # k, b free; logv frozen
    assert infer.free_nparams == 2
    assert infer.dof == 4 - 2


def test_calc_loglike_best_at_truth():
    infer, data, model = _make_infer()
    ll_truth = infer._loglike([2.0, 1.0])
    ll_off = infer._loglike([0.0, 0.0])
    assert ll_truth > ll_off
    assert np.isclose(ll_truth, 0.0, atol=1e-6)


def test_generic_data_properties():
    infer, data, model = _make_infer()
    assert np.allclose(infer.data_x[0], [0, 1, 2, 3])
    assert np.allclose(infer.data_y[0], [1, 3, 5, 7])
    infer.at_par([2.0, 1.0])
    assert np.allclose(infer.model_y[0], [1, 3, 5, 7])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_infer.py -v`
Expected: FAIL — `infer.data_x` does not exist; `_extract` may also fail referencing old props.

- [ ] **Step 3: Replace the spectral property block**

In `curvefit/infer/infer.py`, replace the entire block from the `data_chbin_mean` property through the `model_cts_to_pht` property (currently lines 294-399 — from `    @property\n    def data_chbin_mean(self):` up to and including the end of `def model_cts_to_pht`) with:
```python
    @property
    def data_x(self):
        
        return [unit.x for pair in self.Pair for unit in pair.data.data.values()]
    
    
    @property
    def data_y(self):
        
        return [unit.y for pair in self.Pair for unit in pair.data.data.values()]
    
    
    @property
    def data_xerr(self):
        
        return [unit.xerr for pair in self.Pair for unit in pair.data.data.values()]
    
    
    @property
    def data_yerr(self):
        
        return [unit.yerr for pair in self.Pair for unit in pair.data.data.values()]
    
    
    @property
    def data_up(self):
        
        return [unit.up for pair in self.Pair for unit in pair.data.data.values()]
    
    
    @property
    def model_y(self):
        
        ys = list()
        for pair in self.Pair:
            params = pair.pvalues
            for unit in pair.data.data.values():
                ys.append(pair.mo_func(unit.x, params))
                
        return ys
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_infer.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add curvefit/infer/infer.py tests/test_infer.py
git commit -m "feat: Infer exposes generic x/y/model_y properties"
```

---

### Task 7: Rewrite `util/plot.py` — basic curve plotter

Remove all `spectrum`/`response` imports. Provide `Plot.infer` (data with asymmetric error bars + upper-limit markers, best-fit curve, 1σ posterior band, residual panel), `Plot.model` (a model curve), and `Plot.post_corner` (corner of the posterior). Matplotlib-based.

**Files:**
- Replace: `curvefit/util/plot.py` (entire file)
- Test: `tests/test_plot.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_plot.py`:
```python
import matplotlib
matplotlib.use('Agg')

import numpy as np
from curvefit.model.local import ln
from curvefit.util.plot import Plot


def test_plot_model_returns_figure():
    m = ln()
    m.params['k'].value = 1.0
    m.params['b'].value = 0.0
    fig = Plot.model(m, np.linspace(0, 10, 50))
    assert fig is not None
    assert len(fig.axes) >= 1
```

(A full `Plot.infer`/`Plot.post_corner` smoke test runs in Task 9 against a real `Posterior`.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_plot.py -v`
Expected: FAIL — importing `curvefit.util.plot` raises ImportError on `..data.spectrum`.

- [ ] **Step 3: Replace the file**

Overwrite `curvefit/util/plot.py` entirely with:
```python
import numpy as np
import matplotlib.pyplot as plt



class Plot(object):
    
    @staticmethod
    def model(model, x, ax=None):
        
        x = np.asarray(x, dtype=float)
        y = model.func(x[:, None])
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure
        
        ax.plot(x, y, lw=1.5, label=model.expr)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        
        return fig


    @staticmethod
    def infer(post, nsample=200, ngrid=200, seed=42):
        
        fig, (ax, axr) = plt.subplots(
            2, 1, sharex=True, figsize=(7, 6), 
            gridspec_kw={'height_ratios': [3, 1]})
        
        xmins, xmaxs = list(), list()
        for pair in post.Pair:
            for unit in pair.data.data.values():
                xmins.append(np.min(unit.x))
                xmaxs.append(np.max(unit.x))
        grid = np.linspace(min(xmins), max(xmaxs), ngrid)
        
        post.at_par(post.par_best_ci)
        best = {id(pair): pair.mo_func(grid, pair.pvalues) for pair in post.Pair}
        
        rng = np.random.default_rng(seed)
        samples = post.posterior_sample
        size = min(nsample, samples.shape[0])
        idx = rng.choice(samples.shape[0], size=size, replace=False)
        
        bands = dict()
        for pair in post.Pair:
            curves = list()
            for k in idx:
                post.at_par(samples[k, :post.free_nparams])
                curves.append(pair.mo_func(grid, pair.pvalues))
            bands[id(pair)] = np.quantile(np.array(curves), [0.16, 0.84], axis=0)
        
        post.at_par(post.par_best_ci)
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ci = 0
        for pair in post.Pair:
            color = colors[ci % len(colors)]
            ci += 1
            
            ax.plot(grid, best[id(pair)], color=color, lw=1.5)
            lo, hi = bands[id(pair)]
            ax.fill_between(grid, lo, hi, color=color, alpha=0.3)
            
            for unit in pair.data.data.values():
                norm = ~unit.up
                ax.errorbar(unit.x[norm], unit.y[norm], 
                            yerr=[unit.yerr[0][norm], unit.yerr[1][norm]], 
                            xerr=[unit.xerr[0][norm], unit.xerr[1][norm]], 
                            fmt='o', color=color, ms=3, lw=1, capsize=0)
                
                if np.any(unit.up):
                    ax.errorbar(unit.x[unit.up], unit.y[unit.up], 
                                yerr=0.1 * np.abs(unit.y[unit.up]), 
                                uplims=True, fmt='v', color=color, ms=6)
                
                my = pair.mo_func(unit.x, pair.pvalues)
                sigma = np.where(unit.y < my, unit.yerr[1], unit.yerr[0])
                axr.errorbar(unit.x[norm], ((unit.y - my) / sigma)[norm], 
                             yerr=1, fmt='o', color=color, ms=3, lw=1, capsize=0)
        
        axr.axhline(0, color='gray', lw=1, ls='--')
        ax.set_ylabel('y')
        axr.set_ylabel('residual')
        axr.set_xlabel('x')
        fig.tight_layout()
        
        return fig


    @staticmethod
    def post_corner(post, **kwargs):
        
        import corner
        
        samples = post.posterior_sample[:, :post.free_nparams]
        labels = post.free_plabels
        
        return corner.corner(samples, labels=labels, **kwargs)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_plot.py -v`
Expected: PASS.

- [ ] **Step 5: Delete the now-unused spectral helper**

Run:
```bash
cd /Users/junyang/Documents/python_works/curvefit
git rm curvefit/util/significance.py
```
(Nothing imports it after Task 2. If `git rm` reports it is untracked, use `rm curvefit/util/significance.py`.)

- [ ] **Step 6: Commit**

```bash
git add curvefit/util/plot.py tests/test_plot.py
git commit -m "feat: basic curve Plot (data, best-fit, band, residual, corner)"
```

---

### Task 8: Verify full package import

With `data/__init__.py` and `plot.py` fixed, `import curvefit` and the top-level names must work.

**Files:**
- Test: `tests/test_import_pkg.py`

- [ ] **Step 1: Write the test**

Create `tests/test_import_pkg.py`:
```python
def test_import_top_level():
    import curvefit
    from curvefit import Data, DataUnit, Infer, Plot
    from curvefit.model.local import ln, pl
    assert Data is not None and DataUnit is not None
    assert Infer is not None and Plot is not None
```

- [ ] **Step 2: Run the test**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_import_pkg.py -v`
Expected: PASS. If it fails with a circular-import error, confirm `curvefit/__init__.py` order is `util, data, model, infer` (matching bayspec) and that `util/__init__.py` still imports `Plot`.

- [ ] **Step 3: Confirm `BayesInfer` availability**

`examples` and the spec reference `BayesInfer`. Check whether the codebase exposes it:
Run: `cd /Users/junyang/Documents/python_works/curvefit && python -c "import curvefit; print(hasattr(curvefit, 'BayesInfer'))"`
- If `True`: nothing to do.
- If `False`: `Infer` itself carries `emcee`/`multinest` (see `infer/infer.py`), so add a thin alias. Append to `curvefit/infer/infer.py`:
```python


class BayesInfer(Infer):
    pass
```
and add `BayesInfer` to `curvefit/infer/__init__.py`:
```python
from .pair import Pair
from .infer import Infer, BayesInfer
from .statistic import Statistic
from .posterior import Posterior
```

- [ ] **Step 4: Re-run import test**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_import_pkg.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: full curvefit package imports cleanly"
```

---

### Task 9: End-to-end quickstart + smoke test

Prove the whole pipeline runs: build data → model → `Infer` → `emcee` → `Posterior` → plots.

**Files:**
- Create: `examples/quickstart.py`
- Test: `tests/test_end_to_end.py`

- [ ] **Step 1: Write the end-to-end test**

Create `tests/test_end_to_end.py`:
```python
import matplotlib
matplotlib.use('Agg')

import numpy as np
from curvefit.data.data import Data, DataUnit
from curvefit.model.local import ln
from curvefit.infer.infer import Infer
from curvefit.util.plot import Plot


def test_emcee_recovers_linear(tmp_path):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 30)
    ytrue = 2.0 * x + 1.0
    yerr = np.full(x.size, 0.5)
    y = ytrue + rng.normal(0, 0.5, x.size)

    unit = DataUnit(x, y, yerr=yerr, stat='chi^2')
    data = Data([('d', unit)])
    model = ln()
    model.params['logv'].frozen = True

    infer = Infer([(data, model)])
    post = infer.emcee(nstep=400, discard=100, resume=False, savepath=str(tmp_path))

    k = post.par_best_ci[0]
    b = post.par_best_ci[1]
    assert abs(k - 2.0) < 0.3
    assert abs(b - 1.0) < 1.0

    fig = Plot.infer(post, nsample=50, ngrid=50)
    assert fig is not None
    cfig = Plot.post_corner(post)
    assert cfig is not None
```

- [ ] **Step 2: Run the test to verify it fails (then drives implementation)**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_end_to_end.py -v`
Expected: PASS if Tasks 1-8 are complete. If `Plot.infer` errors on `post.par_best_ci`/`pvalues`, confirm `Posterior` (`infer/posterior.py`) is unchanged and `Pair.pvalues` exists.

- [ ] **Step 3: Write the quickstart example**

Create `examples/quickstart.py`:
```python
import numpy as np
from curvefit import Data, DataUnit, Infer, Plot
from curvefit.model.local import ln

savepath = './quickstart'

rng = np.random.default_rng(0)
x = np.linspace(0, 10, 30)
yerr = np.full(x.size, 0.5)
y = 2.0 * x + 1.0 + rng.normal(0, 0.5, x.size)

unit = DataUnit(x, y, yerr=yerr, stat='chi^2')
data = Data([('d', unit)])
print(data)

model = ln()
model.params['logv'].frozen = True
print(model)

infer = Infer([(data, model)])
print(infer)

post = infer.emcee(nstep=2000, discard=500, resume=False, savepath=savepath)
print(post)

fig = Plot.infer(post)
fig.savefig(f'{savepath}/fit.png', dpi=150)

cfig = Plot.post_corner(post)
cfig.savefig(f'{savepath}/corner.png', dpi=150)
```

- [ ] **Step 4: Run the quickstart end to end**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python examples/quickstart.py`
Expected: prints data/model/infer/posterior tables, samples with a progress bar, writes `quickstart/fit.png` and `quickstart/corner.png` with no errors.

- [ ] **Step 5: Run the full test suite**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest -v`
Expected: all tests across all files PASS.

- [ ] **Step 6: Commit**

```bash
git add examples/quickstart.py tests/test_end_to_end.py
git commit -m "test: end-to-end emcee curve fit + quickstart example"
```

---

## Self-Review

**Spec coverage:**
- §2 file map → Tasks 1-9 cover every file (model fix T1, statistic T2, data T3-4, pair T5, infer T6, plot+significance T7, imports T8, example T9). `util/param|prior|post|info|tools|corner.py` and `infer/posterior.py` are reused untouched (verified generic). ✓
- §3 data layer (x/y/xerr/yerr, asymmetric errors, weight, up, 5 loaders) → Task 3. ✓
- §4 model fix → Task 1. ✓
- §5 six statistics, dict dispatch, formulas verbatim, np.math fix → Task 2. ✓
- §6 model↔statistic bridge (Plan A) → Task 5. ✓
- §7 infer calc_loglike + generic properties (emcee/multinest reused) → Task 6. ✓
- §8 basic plot → Task 7. ✓
- §9 end-to-end quickstart → Task 9. ✓
- §11 validation (`import curvefit`, quickstart runs, six stats selectable, asymmetric/up behavior) → Tasks 2, 8, 9. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. ✓

**Type consistency:** `Statistic._allowed_stats` (T2) consumed by `Pair._allowed_stats` (T5). `Pair.mo_func`/`Pair.pvalues` consumed by `Infer.model_y` (T6) and `Plot.infer` (T7). `Data.exprs`/`.stats`/`.npoints` (T3) consumed by `Infer._extract`/`all_stat` (T6). `DataUnit.x/y/xerr/yerr/weight/up/stat/npoint` (T3) consumed by `Pair._stat_calculate` (T5). Statistic signature `(mo_func, params, x, y, x_err, y_err, w, up)` consistent across T2/T5. ✓

**Known reuse assumptions (verified by reading source):** `infer/posterior.py` uses only `free_par`/`posterior_sample`/`all_stat`/`npoint` — generic, reused as-is. `Infer.emcee/multinest/minimize/_loglike/_prior_transform/_logprior` are unchanged. `Par.value`, `Par.frozen`, `Model.at_par`, `Model.par`, `Model.expr`, `Model.pdicts` exist in current code.
