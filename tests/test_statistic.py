import numpy as np

from curvefit.infer.statistic import Statistic


def lin(x, params):
    return params[0] * np.asarray(x) + params[1]


def test_chi_square_matches_manual():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.1, 2.1, 3.9])
    yerr = np.vstack([np.full(3, 0.1), np.full(3, 0.1)])
    xerr = np.zeros((2, 3))
    w = np.ones(3)
    up = np.zeros(3, dtype=bool)
    params = [2.0, 0.0]
    my = lin(x, params)
    expected = -0.5 * np.sum((y - my) ** 2 / 0.1**2)
    got = Statistic.chi_square(lin, params, x, y, xerr, yerr, w, up)
    assert np.isclose(got, expected)


def test_upper_limit_below_model_is_allowed():
    # an upper-limit point with y >= model contributes 0 (consistent)
    x = np.array([1.0])
    y = np.array([5.0])
    yerr = np.vstack([[0.1], [0.1]])
    xerr = np.zeros((2, 1))
    w = np.ones(1)
    up = np.array([True])
    params = [1.0, 0.0]  # model = 1.0 < y=5.0  -> allowed -> S=0
    got = Statistic.chi_square(lin, params, x, y, xerr, yerr, w, up)
    assert got == 0.0


def test_upper_limit_above_model_is_forbidden():
    x = np.array([1.0])
    y = np.array([0.5])
    yerr = np.vstack([[0.1], [0.1]])
    xerr = np.zeros((2, 1))
    w = np.ones(1)
    up = np.array([True])
    params = [1.0, 0.0]  # model = 1.0 > y=0.5 -> forbidden -> -inf
    got = Statistic.chi_square(lin, params, x, y, xerr, yerr, w, up)
    assert got == -np.inf


def test_all_six_statistics_callable_finite():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.1, 2.0, 3.2])
    yerr = np.vstack([np.full(3, 0.2), np.full(3, 0.2)])
    xerr = np.vstack([np.full(3, 0.1), np.full(3, 0.1)])
    w = np.ones(3)
    up = np.zeros(3, dtype=bool)
    # vdr/odr expect [k, b, logv]; chi^2f expects logv as last
    params = [1.0, 0.0, -1.0]
    for name in ['chi^2', 'chi^2f', 'logchi^2', 'vdr', 'odr', 'groth']:
        func = Statistic._allowed_stats[name]
        val = func(lin, params, x, y, xerr, yerr, w, up)
        assert np.isfinite(val), name
