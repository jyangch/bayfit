from collections import OrderedDict
import os
import warnings

import numpy as np

from ..util.info import Info
from ..util.param import Cfg, Par
from ..util.prior import unif
from ..util.tools import SuperDict, json_dump


class Model:
    _allowed_types = ('add', 'mul', 'math')

    def __init__(self):

        self.expr = 'model'
        self.type = 'add'
        self.comment = 'model base class'

        self.config = OrderedDict()
        self.config['cfg'] = Cfg(0)

        self.params = OrderedDict()
        self.params['par'] = Par(1, unif(0, 2))

    def func(self, X):
        pass

    @property
    def mdicts(self):

        return OrderedDict([(self.expr, self)])

    @property
    def fdicts(self):

        return OrderedDict([(ex, mo.func) for ex, mo in self.mdicts.items()])

    @property
    def cdicts(self):

        return OrderedDict([(ex, mo.config) for ex, mo in self.mdicts.items()])

    @property
    def pdicts(self):

        return OrderedDict([(ex, mo.params) for ex, mo in self.mdicts.items()])

    @property
    def cfg(self):

        cid = 0
        cfg = SuperDict()

        for config in self.cdicts.values():
            for cg in config.values():
                cid += 1
                cfg[str(cid)] = cg

        return cfg

    @property
    def par(self):

        pid = 0
        par = SuperDict()

        for params in self.pdicts.values():
            for pr in params.values():
                pid += 1
                par[str(pid)] = pr

        return par

    @property
    def pvalues(self):

        return tuple([pr.value for pr in self.par.values()])

    @property
    def all_config(self):

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

        all_config = self.all_config.copy()

        return Info.from_list_dict(all_config)

    @property
    def par_info(self):

        all_params = self.all_params.copy()

        for par in all_params:
            if par['Frozen']:
                par['Prior'] = 'frozen'

        all_params = Info.list_dict_to_dict(all_params)

        del all_params['Posterior']
        del all_params['Frozen']

        return Info.from_dict(all_params)

    def save(self, savepath):

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.cfg_info.data_list_dict, savepath + '/model_cfg.json')
        json_dump(self.par_info.data_list_dict, savepath + '/model_par.json')

    @property
    def fit_to(self):

        try:
            return self._fit_to
        except AttributeError:
            raise AttributeError('no data fit to') from None

    @fit_to.setter
    def fit_to(self, new_data):

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

        for i, thi in enumerate(theta):
            self.par[i + 1].val = thi

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

        nsample = max([1 if par.frozen else par.post.nsample for par in self.par.values()])

        return nsample

    @property
    def posterior_sample(self):

        sample = np.vstack(
            [
                np.full(self.posterior_nsample, par.val) if par.frozen else par.post.sample.copy()
                for par in self.par.values()
            ]
        ).T

        return sample

    def sample_statistic(self, sample):

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

        return self.sample_statistic(self.posterior_sample)

    def func_sample(self, X):

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

        return CompositeModel(self, '+', other)

    def __radd__(self, other):

        return self.__add__(other)

    def __sub__(self, other):

        return CompositeModel(self, '-', other)

    def __rsub__(self, other):

        return CompositeModel(other, '-', self)

    def __mul__(self, other):

        return CompositeModel(self, '*', other)

    def __rmul__(self, other):

        return self.__mul__(other)

    def __truediv__(self, other):

        return CompositeModel(self, '/', other)

    def __rtruediv__(self, other):

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


class FrozenConst(Mathematic):
    def __init__(self, value):
        super().__init__()

        self.expr = 'const'
        self.comment = f'constant model with value {value}'

        self.params = OrderedDict()
        self.params['$C$'] = Par(value, frozen=True)

    def func(self, X):

        C = self.params['$C$'].value
        return C


class CompositeModel(Model):
    def __init__(self, m1, op, m2):

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

        return f'({self.m1.expr}{self.op}{self.m2.expr})'

    @property
    def type(self):

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

        return '\n'.join([f'{expr}: {mo.comment}' for expr, mo in self.mdicts.items()])

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

    @property
    def mdicts(self):

        return OrderedDict({**self.m1.mdicts, **self.m2.mdicts})

    @staticmethod
    def _generate_unique_name(name, family, number=2):

        while True:
            new_name = f'{name}_{number}'
            if new_name in family:
                continue
            else:
                break
        return new_name

    @property
    def tdict(self):

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
