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
