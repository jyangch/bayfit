import numpy as np

from ..data.data import Data
from ..model.model import Model
from .statistic import Statistic


class Pair:
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

        return [par.val for par in self.model.par.values()]

    def _stat_calculate(self):

        params = self.pvalues

        loglike = list()
        for unit in self.data.data.values():
            func = self._allowed_stats[unit.stat]
            loglike.append(
                func(
                    self.mo_func, params, unit.x, unit.y, unit.xerr, unit.yerr, unit.weight, unit.up
                )
            )

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
