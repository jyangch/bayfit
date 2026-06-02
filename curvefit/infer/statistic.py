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
