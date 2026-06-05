import numpy as np

from bayfit.infer.statistic import Statistic


def lin(x, params):
    return params[0] * np.asarray(x) + params[1]


def _kw(x, y, xerr, yerr, up, params, lo=None):
    # the statistic interface takes the precomputed model (my) and already
    # normalised f64/bool inputs -- it does no conversion or masking itself,
    # so the caller (here standing in for the data layer) supplies up/lo
    my = np.asarray(lin(x, params), dtype=float)
    n = my.shape[0]
    up = np.zeros(n, dtype=bool) if up is None else np.asarray(up, dtype=bool)
    lo = np.zeros(n, dtype=bool) if lo is None else np.asarray(lo, dtype=bool)
    return dict(
        my=my,
        params=np.asarray(params, dtype=float),
        y=np.asarray(y, dtype=float),
        xerr=np.asarray(xerr, dtype=float),
        yerr=np.asarray(yerr, dtype=float),
        up=up,
        lo=lo,
    )


def test_chi_square_matches_manual():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.1, 2.1, 3.9])
    yerr = np.vstack([np.full(3, 0.1), np.full(3, 0.1)])
    xerr = np.zeros((2, 3))
    up = np.zeros(3, dtype=bool)
    params = [2.0, 0.0]
    my = lin(x, params)
    expected_stat = np.sum((y - my) ** 2 / 0.1**2)
    stat = Statistic.chi_square(**_kw(x, y, xerr, yerr, up, params))['stat']
    assert np.isclose(stat, expected_stat)
    assert np.isclose(-0.5 * stat, -0.5 * expected_stat)


def test_upper_limit_below_model_is_allowed():
    # an upper-limit point with y >= model contributes 0 (consistent)
    x = np.array([1.0])
    y = np.array([5.0])
    yerr = np.vstack([[0.1], [0.1]])
    xerr = np.zeros((2, 1))
    up = np.array([True])
    params = [1.0, 0.0]  # model = 1.0 < y=5.0  -> allowed -> S=0
    out = Statistic.chi_square(**_kw(x, y, xerr, yerr, up, params))
    assert out['stat'] == 0.0
    assert out['residual'][0] == 0.0


def test_upper_limit_above_model_is_forbidden():
    x = np.array([1.0])
    y = np.array([0.5])
    yerr = np.vstack([[0.1], [0.1]])
    xerr = np.zeros((2, 1))
    up = np.array([True])
    params = [1.0, 0.0]  # model = 1.0 > y=0.5 -> forbidden -> stat=inf
    out = Statistic.chi_square(**_kw(x, y, xerr, yerr, up, params))
    assert out['stat'] == np.inf
    assert np.isinf(out['residual'][0])


def test_lower_limit_above_model_is_allowed():
    # a lower-limit point with y <= model contributes 0 (consistent)
    x = np.array([1.0])
    y = np.array([0.5])
    yerr = np.vstack([[0.1], [0.1]])
    xerr = np.zeros((2, 1))
    up = np.zeros(1, dtype=bool)
    lo = np.array([True])
    params = [1.0, 0.0]  # model = 1.0 >= y=0.5 -> allowed -> S=0
    out = Statistic.chi_square(**_kw(x, y, xerr, yerr, up, params, lo=lo))
    assert out['stat'] == 0.0
    assert out['residual'][0] == 0.0


def test_lower_limit_below_model_is_forbidden():
    x = np.array([1.0])
    y = np.array([5.0])
    yerr = np.vstack([[0.1], [0.1]])
    xerr = np.zeros((2, 1))
    up = np.zeros(1, dtype=bool)
    lo = np.array([True])
    params = [1.0, 0.0]  # model = 1.0 < y=5.0 -> forbidden -> stat=inf
    out = Statistic.chi_square(**_kw(x, y, xerr, yerr, up, params, lo=lo))
    assert out['stat'] == np.inf
    assert np.isinf(out['residual'][0])


def test_lower_limit_in_log_space():
    # logchi2 must apply the lower-limit rule in log space
    x = np.array([1.0, 1.0])
    y = np.array([0.5, 5.0])
    yerr = np.vstack([[0.1, 0.1], [0.1, 0.1]])
    xerr = np.zeros((2, 2))
    up = np.zeros(2, dtype=bool)
    lo = np.array([True, True])
    params = [1.0, 0.0]  # model = 1.0; first point allowed, second forbidden
    stat = Statistic.log_chi_square(**_kw(x, y, xerr, yerr, up, params, lo=lo))['stat']
    assert stat == np.inf


def test_lower_limit_defaults_off_when_absent():
    # statistics still work when no lo is supplied (backward compatible)
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.1, 2.1, 3.9])
    yerr = np.vstack([np.full(3, 0.1), np.full(3, 0.1)])
    xerr = np.zeros((2, 3))
    up = np.zeros(3, dtype=bool)
    params = [2.0, 0.0]
    stat = Statistic.chi_square(**_kw(x, y, xerr, yerr, up, params))['stat']
    assert np.isfinite(stat)


def _groth_reference(y, my):
    # high-precision Groth (1975) series via Decimal; valid for moderate counts
    from decimal import Decimal
    from math import factorial

    stat = 0.0
    for yv, myv in zip(y, my, strict=False):
        yi = Decimal(float(yv))
        myi = Decimal(float(myv))
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
        stat += -2.0 * np.log(li)
    return stat


def test_groth_core_matches_reference_moderate():
    from bayfit.infer.statistic import _groth_core

    y = np.array([5.0, 10.0, 3.0, 20.0])
    my = np.array([4.0, 12.0, 2.5, 18.0])
    got = _groth_core(my, y)
    ref = _groth_reference(y, my)
    assert np.isclose(got, ref, rtol=1e-8, atol=1e-8)


def test_groth_core_zero_count_analytic():
    # I0(0) = 1, so a zero count contributes stat = 2 * (y + my)
    from bayfit.infer.statistic import _groth_core

    assert np.isclose(_groth_core(np.array([7.0]), np.array([0.0])), 14.0)
    assert np.isclose(_groth_core(np.array([0.0]), np.array([5.0])), 10.0)


def test_groth_core_finite_for_large_counts():
    from bayfit.infer.statistic import _groth_core

    y = np.array([5000.0, 200.0])
    my = np.array([5000.0, 220.0])
    got = _groth_core(my, y)
    assert np.isfinite(got)


def test_all_six_statistics_callable_finite():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.1, 2.0, 3.2])
    yerr = np.vstack([np.full(3, 0.2), np.full(3, 0.2)])
    xerr = np.vstack([np.full(3, 0.1), np.full(3, 0.1)])
    up = np.zeros(3, dtype=bool)
    # vdr/odr expect [k, b, logv]; chi2f expects logv as last
    params = [1.0, 0.0, -1.0]
    from bayfit.infer.pair import Pair

    for name in ['chi2', 'chi2f', 'logchi2', 'vdr', 'odr', 'groth']:
        func = Pair._allowed_stats[name]
        stat = func(**_kw(x, y, xerr, yerr, up, params))['stat']
        assert np.isfinite(stat), name
