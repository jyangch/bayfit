import numpy as np

from curvefit.infer.statistic import LMFIT_SAFE_STATS, Statistic


def lin(x, params):
    return params[0] * np.asarray(x) + params[1]


def _kwargs(stat, params, up=None):
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.1, 2.1, 3.9, 6.2])
    n = x.shape[0]
    yerr = np.vstack([np.full(n, 0.2), np.full(n, 0.2)])
    xerr = np.vstack([np.full(n, 0.1), np.full(n, 0.1)])
    w = np.ones(n)
    up = np.zeros(n, dtype=bool) if up is None else np.asarray(up, dtype=bool)
    return dict(mo_func=lin, params=params, x=x, y=y, x_err=xerr, y_err=yerr, w=w, up=up)


def test_chi_square_returns_stat_and_residual():
    out = Statistic.chi_square(**_kwargs('chi^2', [2.0, 0.0]))
    assert isinstance(out, tuple) and len(out) == 2
    stat, residual = out
    # chi^2: residual is the signed sqrt of each term, so sum(res**2) == stat
    assert np.isclose(np.sum(residual**2), stat)
    # and equals weighted (y - m)/yerr
    kw = _kwargs('chi^2', [2.0, 0.0])
    my = lin(kw['x'], [2.0, 0.0])
    expected = np.sign(kw['y'] - my) * np.abs(kw['y'] - my) / 0.2
    assert np.allclose(residual, expected)


def test_stat_equals_minus_two_loglike():
    # chi^2 stat must be sum of (y-m)^2/yerr^2
    kw = _kwargs('chi^2', [2.0, 0.0])
    stat, _ = Statistic.chi_square(**kw)
    my = lin(kw['x'], [2.0, 0.0])
    assert np.isclose(stat, np.sum((kw['y'] - my) ** 2 / 0.2**2))


def test_all_six_return_tuple_finite():
    cases = {
        'chi^2': [2.0, 0.0],
        'chi^2f': [2.0, 0.0, -1.0],
        'logchi^2': [2.0, 0.2],
        'vdr': [2.0, 0.0, -1.0],
        'odr': [2.0, 0.0, -1.0],
        'groth': [2.0, 0.1],
    }
    for name, params in cases.items():
        func = Statistic._allowed_stats[name]
        stat, residual = func(**_kwargs(name, params))
        assert np.isfinite(stat), name
        assert residual.shape[0] == 4, name


def test_upper_limit_residual_behavior():
    # one upper-limit point; model below y -> allowed -> residual 0, stat finite
    kw = _kwargs('chi^2', [2.0, 0.0], up=[False, False, False, True])
    # at x=3, model=6.0, y=6.2 -> y>=my allowed -> contribution 0
    stat, residual = Statistic.chi_square(**kw)
    assert residual[3] == 0.0
    assert np.isfinite(stat)


def test_nonfinite_model_returns_inf():
    def bad(x, params):
        return np.full(np.shape(x), np.nan)

    kw = _kwargs('chi^2', [2.0, 0.0])
    kw['mo_func'] = bad
    stat, residual = Statistic.chi_square(**kw)
    assert stat == np.inf
    assert np.all(np.isinf(residual))


def test_lmfit_safe_set():
    assert frozenset({'chi^2', 'logchi^2'}) == LMFIT_SAFE_STATS
