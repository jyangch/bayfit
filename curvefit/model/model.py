from collections import OrderedDict
import warnings

import numpy as np

from ..util.info import Info
from ..util.param import Par
from ..util.prior import unif
from ..util.tools import SuperDict


class Model:
    _allowed_types = ('add', 'mul', 'conv', 'math')

    def __init__(self):

        self.expr = 'model'
        self.comment = 'model base class'

        self.params = OrderedDict()
        self.params['p'] = Par(1, unif(0, 2))

        self.config = OrderedDict()

        self.type = 'add'

    def func(self, X):
        pass

    @property
    def mdicts(self):

        return OrderedDict([(self.expr, self)])

    @property
    def fdicts(self):

        return OrderedDict([(ex, mo.func) for ex, mo in self.mdicts.items()])

    @property
    def pdicts(self):

        return OrderedDict([(ex, mo.params) for ex, mo in self.mdicts.items()])

    @property
    def cdicts(self):

        return OrderedDict([(ex, mo.config) for ex, mo in self.mdicts.items()])

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

        return Info.from_list_dict(self.all_config)

    @property
    def par_info(self):

        par_info = Info.list_dict_to_dict(self.all_params)

        del par_info['Posterior']

        return Info.from_dict(par_info)

    def at_par(self, theta):

        theta = np.array(theta, dtype=float)

        for i, thi in enumerate(theta):
            self.par[i + 1].val = thi

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

        return dict(
            [
                ('mean', mean),
                ('median', median),
                ('Isigma', Isigma),
                ('IIsigma', IIsigma),
                ('IIIsigma', IIIsigma),
            ]
        )

    @property
    def posterior_statistic(self):

        return self.sample_statistic(self.posterior_sample)

    @property
    def par_mean(self):

        return [par.val if par.frozen else par.post.mean for par in self.par.values()]

    @property
    def par_median(self):

        return [par.val if par.frozen else par.post.median for par in self.par.values()]

    @property
    def par_best(self):

        return [par.val if par.frozen else par.post.best for par in self.par.values()]

    @property
    def par_best_ci(self):

        return [par.val if par.frozen else par.post.best_ci for par in self.par.values()]

    def best_func(self, X):

        self.at_par(self.par_best_ci)

        return self.func(X)

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

        print(self.expr)
        print(self.comment)

        print(self.cfg_info.table)
        print(self.par_info.table)

        return ''


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
