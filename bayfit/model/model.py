"""Curve-fitting model base classes and the composition algebra.

``Model`` defines the functional contract (``func(X)``), parameter and config
dictionaries, and posterior-aware evaluators. ``X`` is a 2-D array whose first
column (``X[:,0]``) carries the independent variable. ``Additive``,
``Multiplicative``, and ``Mathematic`` are thin subclasses that fix the
``type`` tag, and ``CompositeModel`` implements the arithmetic algebra
(``+``, ``-``, ``*``, ``/``) used to build multi-component expressions.
"""

from collections import OrderedDict
import os
import warnings

import numpy as np

from ..util.info import Info
from ..util.param import Cfg, Par
from ..util.prior import unif
from ..util.tools import SuperDict, json_dump


class Model:
    """Base class for a curve-fitting model component.

    Subclasses override :meth:`func` with the model's analytical form and
    populate ``config`` (``Cfg`` entries -- frozen values such as fixed
    offsets) and ``params`` (``Par`` entries -- fittable quantities). The
    ``type`` tag -- ``'add'``, ``'mul'``, or ``'math'`` -- drives
    type-checking when models are composed with the arithmetic operators.

    Attributes:
        expr: Short expression label used in tables and composite names.
        type: One of ``'add'``, ``'mul'``, ``'math'``.
        comment: Free-form description of the component.
        config: ``OrderedDict`` of configuration ``Cfg`` entries.
        params: ``OrderedDict`` of fit ``Par`` entries.
    """

    _allowed_types = ('add', 'mul', 'math')

    def __init__(self):
        """Initialise the default dummy component.

        Real subclasses override this to set ``expr``/``type``/``comment``
        and populate ``config``/``params`` with their own entries.
        """

        self.expr = 'model'
        self.type = 'add'
        self.comment = 'model base class'

        self.config = OrderedDict()
        self.config['cfg'] = Cfg(0)

        self.params = OrderedDict()
        self.params['par'] = Par(1, unif(0, 2))

    def func(self, X):
        """Evaluate the model at the input array ``X``.

        ``X`` is a 2-D array where ``X[:,0]`` holds the independent variable
        (e.g., the x-axis grid). Additive models return a y-value array;
        multiplicative and mathematical models return a dimensionless factor.

        Args:
            X: 2-D input array; ``X[:,0]`` is the x-coordinate grid.

        Returns:
            The model value at the given sampling; subclass-specific.
        """

        pass

    @staticmethod
    def _asx(X):
        """Extract the primary x-grid from ``X`` (scalar, 1-D, or 2-D).

        Returns:
            Tuple ``(x, scalar)``: ``x`` is always a 1-D array; ``scalar`` is
            ``True`` when the input was 0-D, signalling the caller to unwrap
            the single result.

        Raises:
            ValueError: If ``X`` has more than two dimensions.
        """

        X = np.asarray(X)
        scalar = X.ndim == 0

        if X.ndim == 0:
            x = X[np.newaxis]
        elif X.ndim == 1:
            x = X
        elif X.ndim == 2:
            x = X[:, 0]
        else:
            raise ValueError('X must be scalar, 1-D, or 2-D')

        return x, scalar

    @property
    def mdicts(self):
        """Mapping from ``expr`` to component model; overridden by composites."""

        return OrderedDict([(self.expr, self)])

    @property
    def fdicts(self):
        """Mapping from ``expr`` to each component's ``func`` callable."""

        return OrderedDict([(ex, mo.func) for ex, mo in self.mdicts.items()])

    @property
    def cdicts(self):
        """Mapping from ``expr`` to each component's ``config`` dict."""

        return OrderedDict([(ex, mo.config) for ex, mo in self.mdicts.items()])

    @property
    def pdicts(self):
        """Mapping from ``expr`` to each component's ``params`` dict."""

        return OrderedDict([(ex, mo.params) for ex, mo in self.mdicts.items()])

    @property
    def cfg(self):
        """Flat :class:`SuperDict` of every config parameter across components."""

        cid = 0
        cfg = SuperDict()

        for config in self.cdicts.values():
            for cg in config.values():
                cid += 1
                cfg[str(cid)] = cg

        return cfg

    @property
    def par(self):
        """Flat :class:`SuperDict` of every fit parameter across components."""

        pid = 0
        par = SuperDict()

        for params in self.pdicts.values():
            for pr in params.values():
                pid += 1
                par[str(pid)] = pr

        return par

    @property
    def pvalues(self):
        """Tuple of current parameter values, preserving ``par`` order."""

        return tuple([pr.value for pr in self.par.values()])

    @property
    def all_config(self):
        """List of per-config rows with component, label, and value."""

        cid = 0
        all_config = list()

        for expr, config in self.cdicts.items():
            for cl, cg in config.items():
                cid += 1

                all_config.append(
                    {'cfg#': str(cid), 'Component': expr, 'Parameter': cl, 'Value': cg.val}
                )

        return all_config

    @property
    def all_params(self):
        """List of per-parameter rows with value, prior, posterior, and frozen flag."""

        pid = 0
        all_params = list()

        for expr, params in self.pdicts.items():
            for pl, pr in params.items():
                pid += 1

                all_params.append(
                    {
                        'par#': str(pid),
                        'Component': expr,
                        'Parameter': pl,
                        'Value': pr.val,
                        'Prior': f'{pr.prior_info}',
                        'Frozen': pr.frozen,
                        'Posterior': f'{pr.post_info}',
                    }
                )

        return all_params

    @property
    def cfg_info(self):
        """Tabular :class:`Info` view of every configuration parameter."""

        all_config = self.all_config.copy()

        return Info.from_list_dict(all_config)

    @property
    def par_info(self):
        """Tabular :class:`Info` view of parameters with frozen ones tagged."""

        all_params = self.all_params.copy()

        for par in all_params:
            if par['Frozen']:
                par['Prior'] = 'frozen'

        all_params = Info.list_dict_to_dict(all_params)

        del all_params['Posterior']
        del all_params['Frozen']

        return Info.from_dict(all_params)

    def save(self, savepath):
        """Dump the config and parameter tables under ``savepath``.

        Args:
            savepath: Directory path. Created if missing.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.cfg_info.data_list_dict, savepath + '/model_cfg.json')
        json_dump(self.par_info.data_list_dict, savepath + '/model_par.json')

    @property
    def fit_to(self):
        """Return the ``Data`` this model is bound to, or raise if unset.

        Raises:
            AttributeError: If no data has been assigned.
        """

        try:
            return self._fit_to
        except AttributeError:
            raise AttributeError('no data fit to') from None

    @fit_to.setter
    def fit_to(self, new_data):
        """Bind a ``Data`` to this model and keep the back-reference in sync.

        Raises:
            ValueError: If ``new_data`` is not a ``Data``.
        """

        from ..data.data import Data

        self._fit_to = new_data

        if not isinstance(self._fit_to, Data):
            raise ValueError('fit_to argument should be Data type!')

        try:
            _ = self._fit_to.fit_with
        except AttributeError:
            self._fit_to.fit_with = self
        else:
            if self._fit_to.fit_with != self:
                self._fit_to.fit_with = self

    def at_par(self, theta):
        """Write every free parameter value from the 1-indexed sequence ``theta``."""

        for i, thi in enumerate(theta):
            self.par[i + 1].val = thi

    @property
    def ys(self):
        """Model evaluated on each bound ``fit_to`` unit's x-grid, as ``float64`` arrays.

        Returns one ``float64`` array per data unit (in ``fit_to`` order),
        computed at the model's current parameters and ready to feed the
        statistic kernels without further conversion.
        """

        return [
            self.func(np.asarray(unit.xs)).astype(float)
            for unit in self.fit_to.data.values()
        ]

    @property
    def ps(self):
        """Parameter vector repeated once per bound ``fit_to`` data unit, as ``float64``.

        Parallels :attr:`ys`: a length-``n_unit`` list whose every element is
        the full ``float64`` parameter vector, aligned with the data units so
        the statistic map receives one ``params`` per unit.
        """

        return [np.array([float(v) for v in self.pvalues])] * len(self.fit_to.data)

    @property
    def par_mean(self):
        """Per-parameter posterior mean (or frozen value when applicable)."""

        return [par.val if par.frozen else par.post.mean for par in self.par.values()]

    @property
    def par_median(self):
        """Per-parameter posterior median (or frozen value)."""

        return [par.val if par.frozen else par.post.median for par in self.par.values()]

    @property
    def par_best(self):
        """Per-parameter posterior best-fit sample (or frozen value)."""

        return [par.val if par.frozen else par.post.best for par in self.par.values()]

    @property
    def par_best_ci(self):
        """Per-parameter best-fit confidence bounds (or frozen value)."""

        return [par.val if par.frozen else par.post.best_ci for par in self.par.values()]

    @property
    def par_truth(self):
        """Per-parameter truth value (or frozen value); may contain ``None``."""

        return [par.val if par.frozen else par.post.truth for par in self.par.values()]

    def mean_func(self, X):
        """Evaluate ``func`` at the posterior mean parameter vector.

        The family ``{mean,median,best,best_ci,truth}_func`` all follow the
        same pattern: set ``par`` from the named posterior summary, then
        call :meth:`func` with the supplied ``X``.
        """

        self.at_par(self.par_mean)

        return self.func(X)

    def median_func(self, X):

        self.at_par(self.par_median)

        return self.func(X)

    def best_func(self, X):

        self.at_par(self.par_best)

        return self.func(X)

    def best_ci_func(self, X):

        self.at_par(self.par_best_ci)

        return self.func(X)

    def truth_func(self, X):

        self.at_par(self.par_truth)

        return self.func(X)

    @property
    def posterior_nsample(self):
        """Number of posterior draws; equals ``1`` when every parameter is frozen."""

        nsample = max([1 if par.frozen else par.post.nsample for par in self.par.values()])

        return nsample

    @property
    def posterior_sample(self):
        """``(nsample, npar)`` matrix of posterior draws.

        Frozen parameters are filled with their fixed value so the matrix
        is rectangular.
        """

        sample = np.vstack(
            [
                np.full(self.posterior_nsample, par.val) if par.frozen else par.post.sample.copy()
                for par in self.par.values()
            ]
        ).T

        return sample

    def sample_statistic(self, sample):
        """Summarize a draw matrix with mean, median, and 1/2/3-sigma intervals.

        Args:
            sample: ``(nsample, ...)`` array of draws.

        Returns:
            Dict with keys ``mean``, ``median``, ``Isigma``, ``IIsigma``,
            ``IIIsigma``, and ``90%``.
        """

        mean = np.mean(sample, axis=0)
        median = np.median(sample, axis=0)

        q = 68.27 / 100
        Isigma = np.quantile(sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        q = 95.45 / 100
        IIsigma = np.quantile(sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        q = 99.73 / 100
        IIIsigma = np.quantile(sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        q = 90 / 100
        ninety_percent = np.quantile(sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        return dict(
            [
                ('mean', mean),
                ('median', median),
                ('Isigma', Isigma),
                ('IIsigma', IIsigma),
                ('IIIsigma', IIIsigma),
                ('90%', ninety_percent),
            ]
        )

    @property
    def par_sample(self):
        """Summary statistics of the posterior parameter matrix."""

        return self.sample_statistic(self.posterior_sample)

    def func_sample(self, X):
        """Return posterior summaries of ``func`` evaluated at ``X``.

        Iterates over every posterior draw, calls :meth:`func`, and
        returns the result of :meth:`sample_statistic` over the collected
        samples.

        Args:
            X: Input array passed to :meth:`func`; may be scalar or array.

        Returns:
            Dict with keys ``mean``, ``median``, ``Isigma``, ``IIsigma``,
            ``IIIsigma``, and ``90%``.
        """

        scalar = np.asarray(X).ndim == 0

        if scalar:
            sample = np.zeros(self.posterior_nsample, dtype=float)
        else:
            sample = np.zeros([self.posterior_nsample, len(X)], dtype=float)

        for i in range(self.posterior_nsample):
            self.at_par(self.posterior_sample[i])
            sample[i] = self.func(X)

        return self.sample_statistic(sample)

    def __add__(self, other):
        """Build a composite that sums this component with ``other``."""

        return CompositeModel(self, '+', other)

    def __radd__(self, other):
        """Right-side variant of :meth:`__add__`."""

        return self.__add__(other)

    def __sub__(self, other):
        """Build a composite that subtracts ``other`` from this component."""

        return CompositeModel(self, '-', other)

    def __rsub__(self, other):
        """Right-side variant of :meth:`__sub__`."""

        return CompositeModel(other, '-', self)

    def __mul__(self, other):
        """Build a composite that multiplies this component by ``other``."""

        return CompositeModel(self, '*', other)

    def __rmul__(self, other):
        """Right-side variant of :meth:`__mul__`."""

        return self.__mul__(other)

    def __truediv__(self, other):
        """Build a composite that divides this component by ``other``."""

        return CompositeModel(self, '/', other)

    def __rtruediv__(self, other):
        """Right-side variant of :meth:`__truediv__`."""

        return CompositeModel(other, '/', self)

    def __str__(self):

        return (
            f'*** Model ***\n'
            f'{self.expr} [{self.type}]\n'
            f'{self.comment}\n'
            f'*** Model Configurations ***\n'
            f'{self.cfg_info.text_table}\n'
            f'*** Model Parameters ***\n'
            f'{self.par_info.text_table}'
        )

    def __repr__(self):

        return self.__str__()

    def _repr_html_(self):

        return (
            f'{self.cfg_info.html_style}'
            f'<details open>'
            f'<summary style="margin-bottom: 10px;"><b>Model</b></summary>'
            f'<p><b>{self.expr} [{self.type}]</b></p>'
            f'<p style="white-space: pre-wrap;">{self.comment}</p>'
            f'<details open style="margin-top: 10px;">'
            f'<summary style="margin-bottom: 10px;"><b>Model Configurations</b></summary>'
            f'{self.cfg_info.html_table}'
            f'</details>'
            f'<details open style="margin-top: 10px;">'
            f'<summary style="margin-bottom: 10px;"><b>Model Parameters</b></summary>'
            f'{self.par_info.html_table}'
            f'</details>'
            f'</details>'
        )


class Additive(Model):
    """Base for additive components; ``type`` is locked to ``'add'``."""

    @property
    def type(self):
        """Model type tag, always ``'add'``."""

        return 'add'

    @type.setter
    def type(self, new_type):
        """No-op; the type tag is fixed by subclassing :class:`Additive`."""

        pass


class Multiplicative(Model):
    """Base for dimensionless multiplicative components; ``type`` is ``'mul'``."""

    @property
    def type(self):
        """Model type tag, always ``'mul'``."""

        return 'mul'

    @type.setter
    def type(self, new_type):
        """No-op; the type tag is fixed by subclassing :class:`Multiplicative`."""

        pass


class Mathematic(Model):
    """Base for dimensionless mathematical components; ``type`` is ``'math'``."""

    @property
    def type(self):
        """Model type tag, always ``'math'``."""

        return 'math'

    @type.setter
    def type(self, new_type):
        """No-op; the type tag is fixed by subclassing :class:`Mathematic`."""

        pass


class FrozenConst(Mathematic):
    """Frozen scalar used to wrap numeric literals in composite expressions."""

    def __init__(self, value):
        """Hold ``value`` as a single frozen parameter.

        Args:
            value: The numeric constant.
        """

        super().__init__()

        self.expr = 'const'
        self.comment = f'constant model with value {value}'

        self.params = OrderedDict()
        self.params['$C$'] = Par(value, frozen=True)

    def func(self, X):
        """Return the stored constant regardless of ``X``."""

        C = self.params['$C$'].value
        return C


class CompositeModel(Model):
    """Binary combination of two models under ``+``/``-``/``*``/``/``.

    The composite's type is inferred from the operand types and the
    operator via :attr:`tdict`; invalid combinations raise ``ValueError``.
    Duplicate component names are made unique with a numeric suffix.
    """

    def __init__(self, m1, op, m2):
        """Wrap two operands and normalize numeric literals to ``FrozenConst``.

        Args:
            m1: Left operand -- a ``Model`` or numeric literal.
            op: One of ``'+'``, ``'-'``, ``'*'``, ``'/'``.
            m2: Right operand -- a ``Model`` or numeric literal.

        Raises:
            ValueError: If either operand has an unsupported type.
        """

        self.op = op

        if isinstance(m1, Model):
            self.m1 = m1
        elif isinstance(m1, (int, float)):
            self.m1 = FrozenConst(m1)
        else:
            raise ValueError(f'unsupported model type for {op}')

        if isinstance(m2, Model):
            self.m2 = m2
        elif isinstance(m2, (int, float)):
            self.m2 = FrozenConst(m2)
        else:
            raise ValueError(f'unsupported model type for {op}')

        for ex in set(self.m1.mdicts.keys()) & set(self.m2.mdicts.keys()):
            if id(self.m1.mdicts[ex]) == id(self.m2.mdicts[ex]):
                msg = f'note that the same object ({ex}) is used multiple times!'
                warnings.warn(msg, stacklevel=2)
            else:
                msg = f'note that the objects with same name ({ex}) are used!'
                warnings.warn(msg, stacklevel=2)

                family = set(self.m1.mdicts.keys()) | set(self.m2.mdicts.keys())
                self.m2.mdicts[ex].expr = self._generate_unique_name(ex, family)

    @property
    def expr(self):
        """Parenthesized expression assembled from the two operands."""

        return f'({self.m1.expr}{self.op}{self.m2.expr})'

    @property
    def type(self):
        """Derived composite type looked up in :attr:`tdict`.

        Raises:
            ValueError: If the operand-type pair is not an allowed combination.
            AssertionError: If either operand has an unknown type tag.
        """

        assert self.m1.type in self._allowed_types, f'unsupported model.type: {self.m1.type}'
        assert self.m2.type in self._allowed_types, f'unsupported model.type: {self.m2.type}'

        type_op = f'{self.m1.type}{self.op}{self.m2.type}'

        if not self.tdict[type_op]:
            msg = f'unsupported model.type {(self.m1.type, self.m2.type)} for {self.op}'
            raise ValueError(msg)
        else:
            return self.tdict[type_op]

    @property
    def comment(self):
        """Concatenated per-component comments, one line per component."""

        return '\n'.join([f'{expr}: {mo.comment}' for expr, mo in self.mdicts.items()])

    def func(self, X):
        """Evaluate the composite by dispatching on :attr:`op`.

        Args:
            X: Input array forwarded to both operands' ``func`` methods.

        Raises:
            ValueError: If ``op`` is not recognized.
        """

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

    @property
    def mdicts(self):
        """Merged component mapping from both operands."""

        return OrderedDict({**self.m1.mdicts, **self.m2.mdicts})

    @staticmethod
    def _generate_unique_name(name, family, number=2):
        """Return ``name`` suffixed with the first integer not already in ``family``."""

        while True:
            new_name = f'{name}_{number}'
            if new_name in family:
                continue
            else:
                break
        return new_name

    @property
    def tdict(self):
        """Lookup table mapping ``'<t1><op><t2>'`` strings to the composite type.

        A value of ``False`` marks an illegal combination.
        """

        return {
            'add+add': 'add',
            'add+mul': False,
            'add+math': 'add',
            'mul+add': False,
            'mul+mul': 'mul',
            'mul+math': 'mul',
            'math+add': 'add',
            'math+mul': 'mul',
            'math+math': 'math',
            'add-add': 'add',
            'add-mul': False,
            'add-math': 'add',
            'mul-add': False,
            'mul-mul': 'mul',
            'mul-math': 'mul',
            'math-add': 'add',
            'math-mul': 'mul',
            'math-math': 'math',
            'add*add': False,
            'add*mul': 'add',
            'add*math': 'add',
            'mul*add': 'add',
            'mul*mul': 'mul',
            'mul*math': 'mul',
            'math*add': 'add',
            'math*mul': 'mul',
            'math*math': 'math',
            'add/add': False,
            'add/mul': 'add',
            'add/math': 'add',
            'mul/add': False,
            'mul/mul': 'mul',
            'mul/math': 'mul',
            'math/add': False,
            'math/mul': 'mul',
            'math/math': 'math',
        }
