import numpy as np

from curvefit.data.data import Data, DataUnit
from curvefit.infer.pair import Pair
from curvefit.model.local import line


def test_loglike_on_exact_linear_data():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    yerr = np.full(4, 0.1)
    unit = DataUnit(x, y, yerr=yerr, stat='chi2')
    data = Data([('d', unit)])

    model = line()
    model.params['k'].val = 2.0
    model.params['b'].val = 1.0
    model.params['logv'].frozen = True  # unused by chi2

    pair = Pair(data, model)
    # model matches data exactly -> chi2 ~ 0 -> loglike ~ 0
    assert np.isclose(pair.loglike, 0.0)
    assert np.isclose(pair.stat, 0.0)
    assert pair.npoint == 4


def test_loglike_penalizes_bad_fit():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    unit = DataUnit(x, y, yerr=np.full(4, 0.1), stat='chi2')
    data = Data([('d', unit)])

    model = line()
    model.params['k'].val = 0.0
    model.params['b'].val = 0.0
    pair = Pair(data, model)
    assert pair.loglike < 0.0


def test_pair_propagates_lower_limit():
    # a lower-limit point whose value sits above the model must forbid the fit
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    unit = DataUnit(
        x,
        y,
        yerr=np.full(4, 0.1),
        up=[False, False, False, False],
        lo=[False, False, False, True],
        stat='chi2',
    )
    data = Data([('d', unit)])

    model = line()
    model.params['k'].val = 0.0
    model.params['b'].val = 0.0  # model = 0 < lower-limit y at every point
    pair = Pair(data, model)
    assert np.isinf(pair.stat)
    assert pair.loglike == -np.inf


def test_variance_stats_reject_model_without_logv():
    import pytest

    from curvefit.model.local import pl

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 4.0, 9.0])
    for bad_stat in ['chi2f', 'vdr', 'odr']:
        unit = DataUnit(x, y, yerr=np.full(3, 0.1), stat=bad_stat)
        data = Data([('d', unit)])
        with pytest.raises(ValueError, match='logv'):
            Pair(data, pl())  # pl has parameters [alpha, logA] -- no logv


def test_variance_stats_accept_ln():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    for stat in ['chi2', 'chi2f', 'logchi2', 'vdr', 'odr', 'groth']:
        unit = DataUnit(x, y, yerr=np.full(3, 0.1), stat=stat)
        data = Data([('d', unit)])
        Pair(data, line())  # line has [k, b, logv] -- all statistics allowed


def test_nonvariance_stats_allow_any_model():
    from curvefit.model.local import pl

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 4.0, 9.0])
    for stat in ['chi2', 'logchi2', 'groth']:
        unit = DataUnit(x, y, yerr=np.full(3, 0.1), stat=stat)
        data = Data([('d', unit)])
        Pair(data, pl())  # no logv needed -> fine with any model


def test_model_ys_and_ps():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 5.0])
    unit = DataUnit(x, y, yerr=np.full(3, 0.1), stat='chi2')
    data = Data([('d', unit)])
    model = line()
    model.params['k'].val = 2.0
    model.params['b'].val = 1.0
    Pair(data, model)  # binds model.fit_to = data so model.ys can evaluate

    # ys: the model evaluated at the data x as float64, fed to the kernels
    assert np.allclose(model.ys[0], [1.0, 3.0, 5.0])
    assert model.ys[0].dtype == np.float64

    # ps: the float64 parameter vector repeated once per data unit
    assert len(model.ps) == 1
    assert np.allclose(model.ps[0], model.pvalues)
    assert model.ps[0].dtype == np.float64


def test_pair_unit_weight_scales_stat():
    # the per-unit weight multiplies that unit's statistic in the total
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.5  # offset from the model below so stat > 0
    model = line()
    model.params['k'].val = 2.0
    model.params['b'].val = 1.0

    base = Pair(Data([('d', DataUnit(x, y, yerr=np.full(4, 0.1), weight=1.0))]), model)
    heavy = Pair(Data([('d', DataUnit(x, y, yerr=np.full(4, 0.1), weight=3.0))]), model)

    assert np.allclose(heavy.weight_list, [3.0])
    assert np.isclose(heavy.stat, 3.0 * base.stat)


def test_pair_nonfinite_model_gives_inf():
    # the non-finite-model guard lives in Pair.stat_func / pseudo_residual_func
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    unit = DataUnit(x, y, yerr=np.full(4, 0.1), stat='chi2')
    data = Data([('d', unit)])
    model = line()
    model.params['k'].val = np.nan  # model evaluates to nan everywhere
    pair = Pair(data, model)

    assert np.isinf(pair.stat)
    assert pair.loglike == -np.inf
    assert np.all(np.isinf(pair.pseudo_residual))


def test_pair_pseudo_residual_matches_stat():
    import numpy as np

    from curvefit.data.data import Data, DataUnit
    from curvefit.infer.pair import Pair
    from curvefit.model.local import line

    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    unit = DataUnit(x, y, yerr=np.full(4, 0.1), stat='chi2')
    data = Data([('d', unit)])
    model = line()
    model.params['k'].val = 2.5
    model.params['b'].val = 0.5
    pair = Pair(data, model)

    pr = pair.pseudo_residual
    # for chi2, sum(pseudo_residual**2) equals the total stat
    assert np.isclose(np.sum(pr**2), pair.stat)
    assert np.isclose(pair.loglike, -0.5 * pair.stat)
