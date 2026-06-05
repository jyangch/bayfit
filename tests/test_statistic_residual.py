import numpy as np

from curvefit.infer.statistic import LMFIT_SAFE_STATS, Statistic


def lin(x, params):
    return params[0] * np.asarray(x) + params[1]


def _kwargs(stat, params, up=None):
    # the statistic interface receives the precomputed model (my) and an f64
    # parameter vector; arrays are already f64/bool (no in-statistic conversion)
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.1, 2.1, 3.9, 6.2])
    n = x.shape[0]
    yerr = np.vstack([np.full(n, 0.2), np.full(n, 0.2)])
    xerr = np.vstack([np.full(n, 0.1), np.full(n, 0.1)])
    up = np.zeros(n, dtype=bool) if up is None else np.asarray(up, dtype=bool)
    lo = np.zeros(n, dtype=bool)
    my = np.asarray(lin(x, params), dtype=float)
    return dict(
        my=my,
        params=np.asarray(params, dtype=float),
        y=y,
        xerr=xerr,
        yerr=yerr,
        up=up,
        lo=lo,
    )


def test_chi_square_returns_stat_and_residual():
    kw = _kwargs('chi2', [2.0, 0.0])
    out = Statistic.chi_square(**kw)
    assert isinstance(out, dict) and 'stat' in out and 'residual' in out
    stat, residual = out['stat'], out['residual']
    # chi2: residual is the signed sqrt of each term, so sum(res**2) == stat
    assert np.isclose(np.sum(residual**2), stat)
    # and equals signed (y - m)/yerr
    my = kw['my']
    expected = np.sign(kw['y'] - my) * np.abs(kw['y'] - my) / 0.2
    assert np.allclose(residual, expected)


def test_stat_equals_minus_two_loglike():
    # chi2 stat must be sum of (y-m)^2/yerr^2
    kw = _kwargs('chi2', [2.0, 0.0])
    stat = Statistic.chi_square(**kw)['stat']
    my = kw['my']
    assert np.isclose(stat, np.sum((kw['y'] - my) ** 2 / 0.2**2))


def test_residual_array_only_for_lmfit_safe_stats():
    # every statistic returns a dict with 'stat'; only the lmfit-safe ones also
    # carry a 'residual' (the others omit it, so .get('residual') is None).
    cases = {
        'chi2': [2.0, 0.0],
        'chi2f': [2.0, 0.0, -1.0],
        'logchi2': [2.0, 0.2],
        'vdr': [2.0, 0.0, -1.0],
        'odr': [2.0, 0.0, -1.0],
        'groth': [2.0, 0.1],
    }
    from curvefit.infer.pair import Pair

    for name, params in cases.items():
        func = Pair._allowed_stats[name]
        out = func(**_kwargs(name, params))
        assert np.isfinite(out['stat']), name
        residual = out.get('residual')
        if name in LMFIT_SAFE_STATS:
            assert residual is not None and residual.shape[0] == 4, name
        else:
            assert residual is None, name


def test_upper_limit_residual_behavior():
    # one upper-limit point; model below y -> allowed -> residual 0, stat finite
    kw = _kwargs('chi2', [2.0, 0.0], up=[False, False, False, True])
    # at x=3, model=6.0, y=6.2 -> y>=my allowed -> contribution 0
    out = Statistic.chi_square(**kw)
    assert out['residual'][3] == 0.0
    assert np.isfinite(out['stat'])


def test_lmfit_safe_set():
    assert frozenset({'chi2', 'logchi2'}) == LMFIT_SAFE_STATS
