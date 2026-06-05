"""Likelihood-statistic kernels for generic curve fitting.

Each statistic receives the model already evaluated on the data x-grid
(``my``, float64), forms the per-point ``-2 * loglike`` contribution ``S``,
and returns a dict ``{'stat': ..., 'residual': ...}`` where
``stat = sum(S)``. For the lmfit-safe statistics (``chi2``, ``logchi2``)
``'residual'`` is the signed square-root of each ``S``, so ``sum(residual**2)
== stat`` and lmfit's least-squares minimises the same objective. The
remaining statistics fit a variance term, so their objective is not a sum of
squares; a least-squares residual is undefined and they omit the ``'residual'``
key (consumers use ``.get('residual')``). Upper-limit points contribute 0 when
consistent and ``inf`` otherwise.

All cores are numba-accelerated explicit loops; ``groth`` evaluates its
Poisson series as ``log(I0(2*sqrt(y*my)))`` in log space (overflow-free).
Every statistic shares one keyword bundle:
``my, params, y, xerr, yerr, up, lo``, where ``my`` is the precomputed
model (float64) and ``params`` is the float64 parameter vector (used by the
variance-fitting kernels to read ``logv``/``k``). No conversion or
normalisation happens inside the statistics -- the caller (the data layer via
:class:`~curvefit.infer.pair.Pair`) supplies every value already as
``float64``/``bool``. ``up``/``lo`` are per-point upper-/lower-limit boolean
masks (normalised by :class:`~curvefit.data.data.DataUnit`; all-``False`` means
no limits): an upper limit is consistent when ``model <= y`` and a lower limit
when ``model >= y``; the inconsistent direction contributes ``inf``.

Attributes:
    LMFIT_SAFE_STATS: Statistic tags whose residual is a true signed
        chi-residual, so lmfit least-squares is exact for them.
"""

import math

import numba as nb
import numpy as np

LMFIT_SAFE_STATS = frozenset({'chi2', 'logchi2'})

LINEAR_ONLY_STATS = frozenset({'chi2f', 'vdr', 'odr'})


@nb.njit(cache=True, fastmath=True)
def _chi_square_core(my, y, yerr, up, lo):
    """Numba kernel for ``chi2``; returns ``(stat, signed_residual)``."""

    n = y.shape[0]
    residual = np.empty(n, dtype=np.float64)
    stat = 0.0

    for i in range(n):
        yi = y[i]
        myi = my[i]

        yei = yerr[1, i] if yi < myi else yerr[0, i]
        si = (yi - myi) ** 2 / yei**2

        if up[i]:
            si = 0.0 if yi >= myi else np.inf
        elif lo[i]:
            si = 0.0 if yi <= myi else np.inf

        stat += si

        sign = 1.0 if yi > myi else (-1.0 if yi < myi else 0.0)
        residual[i] = sign * np.sqrt(si)

    return stat, residual


@nb.njit(cache=True, fastmath=True)
def _log_chi_square_core(my, y, yerr, up, lo):
    """Numba kernel for ``logchi2`` (chi-square in log10 space)."""

    n = y.shape[0]
    residual = np.empty(n, dtype=np.float64)
    stat = 0.0

    for i in range(n):
        logy = np.log10(y[i])
        logmy = np.log10(my[i])

        if logy < logmy:
            logyerr = np.log10(y[i] + yerr[1, i]) - logy
        else:
            logyerr = logy - np.log10(y[i] - yerr[0, i])

        si = (logy - logmy) ** 2 / logyerr**2

        if up[i]:
            si = 0.0 if logy >= logmy else np.inf
        elif lo[i]:
            si = 0.0 if logy <= logmy else np.inf

        stat += si

        sign = 1.0 if logy > logmy else (-1.0 if logy < logmy else 0.0)
        residual[i] = sign * np.sqrt(si)

    return stat, residual


@nb.njit(cache=True, fastmath=True)
def _chi_square_full_core(my, y, yerr, up, lo, logv):
    """Numba kernel for ``chi2f`` (extra log-variance ``logv``); returns ``stat``.

    No residual is produced: the fitted variance adds a ``log(2*pi*sigma^2)``
    term, so the objective is not a sum of squares and a least-squares residual
    is undefined.
    """

    n = y.shape[0]
    stat = 0.0

    for i in range(n):
        yi = y[i]
        myi = my[i]

        yei = yerr[1, i] if yi < myi else yerr[0, i]
        sigma2 = yei**2 + 10.0 ** (2.0 * logv)
        si = (yi - myi) ** 2 / sigma2 + np.log(2.0 * np.pi * sigma2)

        if up[i]:
            si = 0.0 if yi >= myi else np.inf
        elif lo[i]:
            si = 0.0 if yi <= myi else np.inf

        stat += si

    return stat


@nb.njit(cache=True, fastmath=True)
def _vdr_core(my, y, xerr, yerr, up, lo, k, logv):
    """Numba kernel for ``vdr`` (effective-variance regression, linear model).

    Returns ``stat`` only; the fitted variance makes the objective a
    non-sum-of-squares, so no least-squares residual is defined.
    """

    n = y.shape[0]
    stat = 0.0

    for i in range(n):
        yi = y[i]
        myi = my[i]

        yei = yerr[1, i] if yi < myi else yerr[0, i]
        if k > 0:
            xei = xerr[0, i] if yi < myi else xerr[1, i]
        else:
            xei = xerr[1, i] if yi < myi else xerr[0, i]

        sigma2 = np.exp(2.0 * logv) + k**2 * xei**2 + yei**2
        si = (yi - myi) ** 2 / sigma2 + np.log(2.0 * np.pi * sigma2)

        if up[i]:
            si = 0.0 if yi >= myi else np.inf
        elif lo[i]:
            si = 0.0 if yi <= myi else np.inf

        stat += si

    return stat


@nb.njit(cache=True, fastmath=True)
def _odr_core(my, y, xerr, yerr, up, lo, k, logv):
    """Numba kernel for ``odr`` (orthogonal-distance regression, linear model).

    Returns ``stat`` only; the fitted variance makes the objective a
    non-sum-of-squares, so no least-squares residual is defined.
    """

    n = y.shape[0]
    stat = 0.0

    for i in range(n):
        yi = y[i]
        myi = my[i]

        yei = yerr[1, i] if yi < myi else yerr[0, i]
        if k > 0:
            xei = xerr[0, i] if yi < myi else xerr[1, i]
        else:
            xei = xerr[1, i] if yi < myi else xerr[0, i]

        delta2 = (yi - myi) ** 2 / (k**2 + 1.0)
        sigma2 = (k**2 * xei**2 + yei**2) / (k**2 + 1.0) + np.exp(2.0 * logv)
        si = delta2 / sigma2 + np.log(2.0 * np.pi * sigma2)

        if up[i]:
            si = 0.0 if yi >= myi else np.inf
        elif lo[i]:
            si = 0.0 if yi <= myi else np.inf

        stat += si

    return stat


@nb.njit(cache=True, fastmath=True)
def _log_i0_series(z):
    """Return ``log(I0(2*sqrt(z)))`` for ``z >= 0`` via a stable log-sum-exp.

    Evaluates ``sum_m z**m / (m!)**2`` (which equals the modified Bessel
    function ``I0(2*sqrt(z))``) in log space, so it never overflows even when
    the individual terms are astronomically large.
    """

    if z <= 0.0:
        return 0.0

    log_z = np.log(z)
    sqrt_z = np.sqrt(z)

    max_log = 0.0  # m = 0 term: 0 * log_z - 2 * lgamma(1) = 0
    sum_exp = 1.0

    m = 1
    while True:
        log_term = m * log_z - 2.0 * math.lgamma(m + 1.0)

        if log_term > max_log:
            sum_exp = sum_exp * np.exp(max_log - log_term) + 1.0
            max_log = log_term
        else:
            sum_exp += np.exp(log_term - max_log)

        if m > sqrt_z and log_term < max_log - 50.0:
            break

        m += 1

    return max_log + np.log(sum_exp)


@nb.njit(cache=True, fastmath=True)
def _groth_core(my, y):
    """Numba kernel for the Groth exact-Poisson statistic; returns ``stat``.

    Uses the closed form ``sum_m (y*my)**m / (m!)**2 == I0(2*sqrt(y*my))`` and
    evaluates ``log(I0)`` in log space (:func:`_log_i0_series`), giving
    ``stat = sum_i 2*(y_i + my_i) - 2*log(I0(2*sqrt(y_i*my_i)))``. No residual
    is produced (the objective is not a sum of squares). Numerically stable for
    large counts, where the older ``Decimal`` series overflowed on conversion.
    """

    n = y.shape[0]
    stat = 0.0

    for i in range(n):
        yi = y[i]
        myi = my[i]

        ln_s = _log_i0_series(yi * myi)
        stat += 2.0 * (yi + myi) - 2.0 * ln_s

    return stat


class Statistic:
    """Curve-fit statistic dispatch table returning ``{'stat', 'residual'}`` dicts.

    Every method takes the shared keyword bundle ``my, params, y, xerr,
    yerr, up, lo``, where ``my`` is the model already evaluated on the
    unit's x-grid (``float64``) and ``params`` is the model's ``float64``
    parameter vector (used by the variance-fitting statistics to read ``logv``
    and ``k``). Inputs are assumed to be ``float64``/``bool`` already, so no
    conversion happens here; the non-finite-``my`` guard lives in
    :class:`~curvefit.infer.pair.Pair`'s ``stat_func``/``pseudo_residual_func``.

    Each method returns a dict with ``'stat'`` (the summed ``-2 * loglike``) and,
    for the lmfit-safe statistics (``chi2``, ``logchi2``), a ``'residual'`` —
    the signed per-point square-root used by lmfit. The variance-fitting
    statistics omit ``'residual'`` (consumers use ``.get('residual')``). The
    tag-to-method dispatch table lives on
    :class:`~curvefit.infer.pair.Pair` as ``_allowed_stats``.
    """

    @staticmethod
    def chi_square(**kwargs):
        """Gaussian chi-square with fixed per-point errors."""

        stat, residual = _chi_square_core(
            kwargs['my'], kwargs['y'], kwargs['yerr'], kwargs['up'], kwargs['lo']
        )
        return {'stat': stat, 'residual': residual}

    @staticmethod
    def log_chi_square(**kwargs):
        """Chi-square evaluated in log10 space."""

        stat, residual = _log_chi_square_core(
            kwargs['my'], kwargs['y'], kwargs['yerr'], kwargs['up'], kwargs['lo']
        )
        return {'stat': stat, 'residual': residual}

    @staticmethod
    def chi_square_full(**kwargs):
        """Gaussian chi-square with an extra fitted log-variance ``logv``.

        Returns ``{'stat': stat}`` (no ``'residual'``); the fitted variance
        makes the objective a non-sum-of-squares, so a residual is undefined.
        """

        params = kwargs['params']

        stat = _chi_square_full_core(
            kwargs['my'],
            kwargs['y'],
            kwargs['yerr'],
            kwargs['up'],
            kwargs['lo'],
            params[-1],
        )
        return {'stat': stat}

    @staticmethod
    def vdr(**kwargs):
        """Effective-variance regression for a linear model ``[k, b, logv]``.

        Returns ``{'stat': stat}`` (no ``'residual'``); the fitted variance
        makes the objective a non-sum-of-squares, so a residual is undefined.
        """

        params = kwargs['params']

        stat = _vdr_core(
            kwargs['my'],
            kwargs['y'],
            kwargs['xerr'],
            kwargs['yerr'],
            kwargs['up'],
            kwargs['lo'],
            params[0],
            params[2],
        )
        return {'stat': stat}

    @staticmethod
    def odr(**kwargs):
        """Orthogonal-distance regression for a linear model ``[k, b, logv]``.

        Returns ``{'stat': stat}`` (no ``'residual'``); the fitted variance
        makes the objective a non-sum-of-squares, so a residual is undefined.
        """

        params = kwargs['params']

        stat = _odr_core(
            kwargs['my'],
            kwargs['y'],
            kwargs['xerr'],
            kwargs['yerr'],
            kwargs['up'],
            kwargs['lo'],
            params[0],
            params[2],
        )
        return {'stat': stat}

    @staticmethod
    def groth(**kwargs):
        """Groth exact-Poisson statistic; ``up`` and ``lo`` are ignored.

        Returns ``{'stat': stat}`` (no ``'residual'``); the objective is not a
        sum of squares, so a residual is undefined.
        """

        return {'stat': _groth_core(kwargs['my'], kwargs['y'])}
