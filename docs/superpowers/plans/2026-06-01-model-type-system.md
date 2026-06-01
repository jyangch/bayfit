# Model Type System + local Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give curvefit's model layer bayspec's four-type system (`add`/`mul`/`conv`/`math`) via `Additive`/`Multiplicative`/`Mathematic` base classes and strict `CompositeModel` type validation, and reorganize `model/local.py` into a `model/local/` package (`additive.py`/`multiplicative.py`/`mathematic.py`).

**Architecture:** Pure-increment change to `curvefit/model/` only. Three base classes lock `type`; `CompositeModel` infers the composite type from a `tdict` lookup and raises on illegal combinations (lazy, on `.type` access — mirrors bayspec). The 8 existing models become `Additive`; a `const` (`Mathematic`) and `expcut` (`Multiplicative`) seed the other two files. Math of existing models is unchanged and guarded by `tests/test_local_models.py`.

**Tech Stack:** Python 3.12, numpy, scipy, pytest, ruff.

**Conventions (do not violate):**
- NEVER modify the sibling `bayspec` package (read-only reference at `/Users/junyang/Documents/python_works/bayspec`).
- Match bayspec/curvefit style: `OrderedDict`, property/setter pattern, no type hints, ruff-clean (`E,W,F,I,UP,B,SIM,RUF`, single quotes, one blank line between methods).
- Model input is a 2-D array `X`; `X[:, 0]` is x. Model funcs are `func(self, X)`.
- `Par.val` is the settable raw value; `Par.value` is read-only (scale/unit transform). Tests set values via `.val`.
- After every code change run `ruff check . && ruff format --check . && python -m pytest -q` and only commit when all pass. Run pytest as its own command (don't pipe through `tail`).

---

### Task 1: Type base classes in `model.py`

Add `Additive`/`Multiplicative`/`Mathematic` base classes (each locks `type`), set a default `self.type = 'add'` on the base `Model`, and reparent `FrozenConst` to `Mathematic`.

**Files:**
- Modify: `curvefit/model/model.py`
- Test: `tests/test_model_types.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_types.py`:
```python
import numpy as np

from curvefit.model.model import Additive, Mathematic, Model, Multiplicative


def test_base_classes_lock_type():
    class A(Additive):
        pass

    class M(Multiplicative):
        pass

    class H(Mathematic):
        pass

    assert A().type == 'add'
    assert M().type == 'mul'
    assert H().type == 'math'


def test_type_setter_is_noop_on_subclasses():
    class A(Additive):
        pass

    a = A()
    a.type = 'mul'   # locked; setter is a no-op
    assert a.type == 'add'


def test_base_model_default_type():
    assert Model().type == 'add'
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_model_types.py -v`
Expected: ImportError/AttributeError — `Additive`/`Multiplicative`/`Mathematic` don't exist and `Model().type` is unset.

- [ ] **Step 3: Set the base default type**

In `curvefit/model/model.py`, in `Model.__init__`, add `self.type = 'add'` after the `self.config = OrderedDict()` line:
```python
        self.config = OrderedDict()

        self.type = 'add'
```

- [ ] **Step 4: Add the three base classes**

In `curvefit/model/model.py`, insert these three classes immediately BEFORE the `class FrozenConst` definition:
```python
class Additive(Model):
    @property
    def type(self):

        return 'add'

    @type.setter
    def type(self, new_type):

        pass


class Multiplicative(Model):
    @property
    def type(self):

        return 'mul'

    @type.setter
    def type(self, new_type):

        pass


class Mathematic(Model):
    @property
    def type(self):

        return 'math'

    @type.setter
    def type(self, new_type):

        pass
```

- [ ] **Step 5: Reparent `FrozenConst` to `Mathematic`**

In `curvefit/model/model.py`, change the `FrozenConst` class declaration line from:
```python
class FrozenConst(Model):
```
to:
```python
class FrozenConst(Mathematic):
```
(Leave the rest of `FrozenConst` unchanged.)

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_model_types.py -v`
Expected: 3 PASS. Note: `Model().type` works because base `__init__` sets `self.type = 'add'`; subclass instances ignore that assignment via the no-op setter.

- [ ] **Step 7: Gate + commit**

```bash
cd /Users/junyang/Documents/python_works/curvefit
ruff check . && ruff format --check . && python -m pytest -q
git add curvefit/model/model.py tests/test_model_types.py
git commit -m "feat: add Additive/Multiplicative/Mathematic model base classes"
```
Expected: ruff clean, all tests pass.

---

### Task 2: Strict composite type validation in `CompositeModel`

Add a `tdict` lookup and a `type` property to `CompositeModel` so the composite type is inferred and illegal combinations raise `ValueError` (lazy, on `.type` access — mirrors bayspec). `func` is unchanged (`+ - * /` only; no `()` operator).

**Files:**
- Modify: `curvefit/model/model.py` (`CompositeModel`)
- Test: `tests/test_model_types.py` (append)

- [ ] **Step 1: Write the failing tests (append)**

Append to `tests/test_model_types.py`:
```python
def test_composite_type_add_plus_add():
    from curvefit.model.local import ln, pl

    assert (ln() + pl()).type == 'add'


def test_composite_type_add_times_mul():
    from curvefit.model.local import expcut, pl

    assert (pl() * expcut()).type == 'add'


def test_illegal_composite_raises():
    import pytest

    from curvefit.model.local import ln, pl

    with pytest.raises(ValueError):
        _ = (ln() * pl()).type   # add * add is illegal


def test_const_plus_add_is_add():
    from curvefit.model.local import const, ln

    assert (ln() + const()).type == 'add'
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_model_types.py -k composite -v`
Expected: FAIL — `CompositeModel` has no `type` property yet (and `local.expcut`/`local.const` are added in Task 3; these specific tests pass after Task 3). The `test_illegal_composite_raises` will also be unsatisfied until the `type` property exists.

- [ ] **Step 3: Add `tdict` and `type` to `CompositeModel`**

In `curvefit/model/model.py`, inside `class CompositeModel`, add the following two members (place `tdict` as a property and `type` as a property; put them right after the existing `expr` property):
```python
    @property
    def tdict(self):

        return {
            'add+add': 'add', 'add+mul': False, 'add+math': 'add',
            'mul+add': False, 'mul+mul': 'mul', 'mul+math': 'mul',
            'math+add': 'add', 'math+mul': 'mul', 'math+math': 'math',
            'add-add': 'add', 'add-mul': False, 'add-math': 'add',
            'mul-add': False, 'mul-mul': 'mul', 'mul-math': 'mul',
            'math-add': 'add', 'math-mul': 'mul', 'math-math': 'math',
            'add*add': False, 'add*mul': 'add', 'add*math': 'add',
            'mul*add': 'add', 'mul*mul': 'mul', 'mul*math': 'mul',
            'math*add': 'add', 'math*mul': 'mul', 'math*math': 'math',
            'add/add': False, 'add/mul': 'add', 'add/math': 'add',
            'mul/add': False, 'mul/mul': 'mul', 'mul/math': 'mul',
            'math/add': False, 'math/mul': 'mul', 'math/math': 'math',
        }

    @property
    def type(self):

        assert self.m1.type in self._allowed_types, f'unsupported model.type: {self.m1.type}'
        assert self.m2.type in self._allowed_types, f'unsupported model.type: {self.m2.type}'

        type_op = f'{self.m1.type}{self.op}{self.m2.type}'

        if not self.tdict.get(type_op, False):
            msg = f'unsupported model.type {(self.m1.type, self.m2.type)} for {self.op}'
            raise ValueError(msg)
        else:
            return self.tdict[type_op]
```

- [ ] **Step 4: Run (these pass once Task 3 adds const/expcut)**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_model_types.py -k "illegal or add_plus_add" -v`
Expected: `test_illegal_composite_raises` and `test_composite_type_add_plus_add` PASS now. The `const`/`expcut` tests pass after Task 3.

- [ ] **Step 5: Gate + commit**

```bash
cd /Users/junyang/Documents/python_works/curvefit
ruff check . && ruff format --check . && python -m pytest -q -k "not const_plus and not add_times_mul"
git add curvefit/model/model.py tests/test_model_types.py
git commit -m "feat: CompositeModel strict type validation via tdict"
```
Expected: ruff clean; the two tests needing const/expcut are deselected here and covered after Task 3.

---

### Task 3: Reorganize `model/local.py` into a `model/local/` package

Split the single module into `additive.py` (the 8 existing models, reparented to `Additive`), plus `mathematic.py` (`const`) and `multiplicative.py` (`expcut`), with an `__init__.py` that auto-collects `local_models`. Remove `local.py`.

**Files:**
- Create: `curvefit/model/local/__init__.py`
- Create: `curvefit/model/local/additive.py`
- Create: `curvefit/model/local/mathematic.py`
- Create: `curvefit/model/local/multiplicative.py`
- Delete: `curvefit/model/local.py`
- Test: `tests/test_local_package.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_local_package.py`:
```python
import numpy as np

from curvefit.model.local import const, expcut, list_local_models, ln, pl


def test_all_models_registered():
    names = set(list_local_models())
    expected = {'ln', 'pl', 'expd', 'bln', 'bpl', 'sbpl', 'spindown', 'psd',
                'const', 'expcut'}
    assert expected <= names


def test_const_returns_constant_array():
    m = const()
    m.params['C'].val = 3.5
    X = np.array([1.0, 2.0, 3.0])[:, None]
    y = m.func(X)
    assert np.allclose(y, [3.5, 3.5, 3.5])
    assert m.type == 'math'


def test_expcut_is_unit_at_zero_and_decreasing():
    m = expcut()
    m.params['logxc'].val = 0.0   # xc = 1
    X = np.array([0.0, 1.0, 2.0])[:, None]
    y = m.func(X)
    assert np.isclose(y[0], 1.0)
    assert y[1] < y[0] and y[2] < y[1]
    assert m.type == 'mul'


def test_existing_models_are_additive():
    assert ln().type == 'add'
    assert pl().type == 'add'
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/junyang/Documents/python_works/curvefit && python -m pytest tests/test_local_package.py -v`
Expected: ImportError — `const`/`expcut`/`list_local_models` not importable yet (or `local` is still a module).

- [ ] **Step 3: Move `local.py` into the package as `additive.py`**

```bash
cd /Users/junyang/Documents/python_works/curvefit
mkdir curvefit/model/local
git mv curvefit/model/local.py curvefit/model/local/additive.py
```

- [ ] **Step 4: Edit `additive.py` header and base classes**

In `curvefit/model/local/additive.py`:
1. Replace the import block at the top (currently):
```python
from collections import OrderedDict
from os.path import abspath, dirname

import numpy as np
from scipy.integrate import solve_ivp

from ..util.param import Par
from ..util.prior import unif
from .model import Model

docs_path = dirname(dirname(dirname(abspath(__file__)))) + '/docs'
```
with (deeper relative imports; drop the unused `docs_path`/`os.path`):
```python
from collections import OrderedDict

import numpy as np
from scipy.integrate import solve_ivp

from ...util.param import Par
from ...util.prior import unif
from ..model import Additive
```
2. Change every model class declaration `class NAME(Model):` to `class NAME(Additive):` for all 8 models: `ln`, `pl`, `expd`, `bln`, `bpl`, `sbpl`, `spindown`, `psd`. (Use replace-all of `(Model):` → `(Additive):`.)
3. Remove the trailing `local_models = {...}`, `list_local_models`, and `__all__` block at the bottom of the file IF present (that responsibility moves to the package `__init__.py`). Keep only the model class definitions.

Do NOT change any `func` body — the math must stay identical (guarded by `tests/test_local_models.py`).

- [ ] **Step 5: Create `mathematic.py`**

Create `curvefit/model/local/mathematic.py`:
```python
from collections import OrderedDict

import numpy as np

from ...util.param import Par
from ...util.prior import unif
from ..model import Mathematic


class const(Mathematic):
    def __init__(self):
        super().__init__()

        self.expr = 'const'
        self.comment = 'constant model'

        self.params = OrderedDict()
        self.params['C'] = Par(1, unif(-10, 10))

    def func(self, X):

        x = X[:, 0]

        C = self.params['C'].value

        return C * np.ones_like(x)
```

- [ ] **Step 6: Create `multiplicative.py`**

Create `curvefit/model/local/multiplicative.py`:
```python
from collections import OrderedDict

import numpy as np

from ...util.param import Par
from ...util.prior import unif
from ..model import Multiplicative


class expcut(Multiplicative):
    def __init__(self):
        super().__init__()

        self.expr = 'expcut'
        self.comment = 'exponential cutoff model'

        self.params = OrderedDict()
        self.params['logxc'] = Par(0, unif(-10, 10))

    def func(self, X):

        x = X[:, 0]

        xc = 10 ** self.params['logxc'].value

        return np.exp(-x / xc)
```

- [ ] **Step 7: Create the package `__init__.py`**

Create `curvefit/model/local/__init__.py`:
```python
from .additive import *  # noqa: F403
from .mathematic import *  # noqa: F403
from .multiplicative import *  # noqa: F403
from ..model import Model

local_models = {
    name: cls
    for name, cls in globals().items()
    if isinstance(cls, type)
    and issubclass(cls, Model)
    and name not in ['Model', 'Additive', 'Multiplicative', 'Mathematic']
}


def list_local_models():

    return list(local_models.keys())


__all__ = [*list(local_models.keys()), 'list_local_models', 'local_models']
```

- [ ] **Step 8: Run the new + golden + type tests**

Run:
```bash
cd /Users/junyang/Documents/python_works/curvefit
python -m pytest tests/test_local_package.py tests/test_local_models.py tests/test_model_types.py -v
```
Expected: ALL pass. `tests/test_local_models.py` (golden outputs of the 8 moved models) confirms the math is unchanged; `test_model_types.py` const/expcut composite tests now pass too.

- [ ] **Step 9: Full gate**

Run:
```bash
cd /Users/junyang/Documents/python_works/curvefit
ruff check . ; ruff format --check . ; python -m pytest -q
```
Expected: `All checks passed!`, all files formatted, all tests pass. If ruff flags the new files (e.g., import order), fix per its autofix (`ruff check --fix . && ruff format .`) and re-run.

- [ ] **Step 10: Commit**

```bash
cd /Users/junyang/Documents/python_works/curvefit
git add -A
git commit -m "refactor: split model/local into additive/multiplicative/mathematic package"
```

---

### Task 4: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Confirm public imports and registry**

Run:
```bash
cd /Users/junyang/Documents/python_works/curvefit
python -c "
import curvefit
from curvefit.model.local import ln, pl, expd, bln, bpl, sbpl, spindown, psd, const, expcut, list_local_models
print(sorted(list_local_models()))
from curvefit.model.model import Additive, Multiplicative, Mathematic
print('types:', ln().type, const().type, expcut().type)
print('composite add+add:', (ln()+pl()).type)
"
```
Expected: prints the 10 model names, `types: add math mul`, `composite add+add: add`, no errors.

- [ ] **Step 2: Full suite + ruff once more**

Run:
```bash
cd /Users/junyang/Documents/python_works/curvefit
ruff check . ; ruff format --check . ; python -m pytest -q
```
Expected: clean + all tests pass.

---

## Self-Review

**Spec coverage:** §2 type system → Tasks 1-2 (base classes, FrozenConst→Mathematic, base default type, tdict/type validation). §3 local package → Task 3 (additive moves 8 models, mathematic `const`, multiplicative `expcut`, `__init__` registry; local.py removed; model/__init__ unchanged — still `from .local import *`). §4 validation → golden test (untouched) + `test_model_types.py` + `test_local_package.py` + Task 4. §5 scope (no `()`/conv models) → honored: `func` unchanged, tdict has only add/mul/math rows. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; the additive.py move uses explicit edit instructions + replace-all rule rather than reproducing 600 lines (math guarded by the golden test). ✓

**Type consistency:** `Additive/Multiplicative/Mathematic` defined in Task 1 are imported by Task 3's local files and the registry exclusion list. `const` exposes `params['C']`; `expcut` exposes `params['logxc']` — matched in tests. `tdict` keys use the exact `'<t1><op><t2>'` form built by the `type` property. `list_local_models`/`local_models` names match between `__init__.py` and tests. ✓

**Note on `model/__init__.py`:** it already does `from .local import *  # noqa: F403`; since `local` becomes a package whose `__init__` re-exports the same names, no change is needed there.
