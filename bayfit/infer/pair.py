"""Binding between a :class:`Data` and a :class:`Model` for fit evaluation.

Holds one model-data pair, evaluates each unit's statistic by dispatching
on its tag, and publishes the total statistic, per-bin pseudo-residuals
(for lmfit), sigma residuals (for plots), and the log-likelihood.
"""

from types import MappingProxyType

import numpy as np

from ..data.data import Data
from ..model.model import Model
from .statistic import LINEAR_ONLY_STATS, Statistic


class Pair:
    """Binds one ``Data`` and one ``Model`` and evaluates their joint statistic.

    Dispatches on each unit's ``stat`` tag via :attr:`_allowed_stats`; every
    statistic returns ``(stat, residual)``.

    Attributes:
        data: The bound :class:`~bayfit.data.data.Data`.
        model: The bound :class:`~bayfit.model.model.Model`.
    """

    _allowed_stats = MappingProxyType(
        {
            'chi2': Statistic.chi_square,
            'chi2f': Statistic.chi_square_full,
            'logchi2': Statistic.log_chi_square,
            'vdr': Statistic.vdr,
            'odr': Statistic.odr,
            'groth': Statistic.groth,
        }
    )

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

        self._check_stats_compatible()

    def _check_stats_compatible(self):
        """Reject variance-fitting statistics on models other than ``line``.

        ``chi2f``/``vdr``/``odr`` read a trailing ``logv`` (and ``vdr``/``odr``
        assume the linear ``[k, b, logv]`` layout), so they are only valid for
        the :class:`~bayfit.model.local.line` model.

        Raises:
            ValueError: If any unit uses such a statistic with a non-``line`` model.
        """

        from ..model.local import line

        for unit in self.data.data.values():
            stat = unit.stat

            if stat in LINEAR_ONLY_STATS and not isinstance(self.model, line):
                raise ValueError(
                    f"statistic '{stat}' is only valid for the linear 'line' model "
                    f"[k, b, logv]; model '{self.model.expr}' is not a line model"
                )

    @property
    def residuals(self):
        """Per-unit sigma residuals ``(y - model) / yerr`` for diagnostics/plots.

        ``yerr`` is the asymmetric ``[low, high]`` error selected per point by
        the sign of ``y - model``.
        """

        return list(
            map(
                lambda yi, mi, ei: (yi - mi) / np.where(yi < mi, ei[1], ei[0]),
                self.data.ys,
                self.model.ys,
                self.data.yerr,
            )
        )

    @property
    def stat_func(self):
        """Closure returning the per-unit statistic; ``+inf`` when ``my`` is non-finite."""

        return lambda my, params, y, xerr, yerr, up, lo, stat: (
            np.inf
            if not np.isfinite(my).all()
            else self._allowed_stats[stat](
                my=my, params=params, y=y, xerr=xerr, yerr=yerr, up=up, lo=lo
            ).get('stat')
        )

    @property
    def pseudo_residual_func(self):
        """Closure returning the per-unit pseudo-residual; ``inf`` array when ``my`` is non-finite."""

        return lambda my, params, y, xerr, yerr, up, lo, stat: (
            np.ones_like(my) * np.inf
            if not np.isfinite(my).all()
            else self._allowed_stats[stat](
                my=my, params=params, y=y, xerr=xerr, yerr=yerr, up=up, lo=lo
            ).get('residual')
        )

    def _stat_calculate(self):

        return np.array(
            list(
                map(
                    self.stat_func,
                    self.model.ys,
                    self.model.ps,
                    self.data.ys,
                    self.data.xerr,
                    self.data.yerr,
                    self.data.ups,
                    self.data.los,
                    self.data.stats,
                )
            ),
            dtype=float,
        )

    def _pseudo_residual_calculate(self):

        return list(
            map(
                self.pseudo_residual_func,
                self.model.ys,
                self.model.ps,
                self.data.ys,
                self.data.xerr,
                self.data.yerr,
                self.data.ups,
                self.data.los,
                self.data.stats,
            )
        )

    @property
    def stat_list(self):
        """Per-unit statistic values."""

        return np.array(self._stat_calculate())

    @property
    def pseudo_residual_list(self):
        """Per-unit signed pseudo-residual arrays."""

        return self._pseudo_residual_calculate()

    @property
    def weight_list(self):
        """Per-unit scalar weights applied to each unit's statistic."""

        return self.data.weights

    @property
    def stat(self):
        """Total weighted statistic summed across units."""

        return np.sum(self.stat_list * self.weight_list)

    @property
    def pseudo_residual(self):
        """Pseudo-residual vector concatenated across units (lmfit target)."""

        return np.concatenate(
            [
                rd * np.sqrt(wt)
                for rd, wt in zip(self.pseudo_residual_list, self.weight_list, strict=False)
            ]
        )

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
