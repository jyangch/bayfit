import numpy as np

from curvefit.data.data import Data, DataUnit
from curvefit.infer.infer import Infer
from curvefit.model.local import ln


def _make_infer():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    unit = DataUnit(x, y, yerr=np.full(4, 0.1), stat='chi^2')
    data = Data([('d', unit)])
    model = ln()
    model.params['logv'].frozen = True
    return Infer([(data, model)]), data, model


def test_free_params_and_dof():
    infer, _data, _model = _make_infer()
    # k, b free; logv frozen
    assert infer.free_nparams == 2
    assert infer.dof == 4 - 2


def test_calc_loglike_best_at_truth():
    infer, _data, _model = _make_infer()
    ll_truth = infer._loglike([2.0, 1.0])
    ll_off = infer._loglike([0.0, 0.0])
    assert ll_truth > ll_off
    assert np.isclose(ll_truth, 0.0, atol=1e-6)


def test_generic_data_properties():
    infer, _data, _model = _make_infer()
    assert np.allclose(infer.data_x[0], [0, 1, 2, 3])
    assert np.allclose(infer.data_y[0], [1, 3, 5, 7])
    infer.at_par([2.0, 1.0])
    assert np.allclose(infer.model_y[0], [1, 3, 5, 7])


def test_calc_stat_pseudo_residual_and_clean_labels():
    import numpy as np

    from curvefit.data.data import Data, DataUnit
    from curvefit.infer.infer import Infer
    from curvefit.model.local import ln

    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    unit = DataUnit(x, y, yerr=np.full(4, 0.1), stat='chi^2')
    data = Data([('d', unit)])
    model = ln()
    model.params['logv'].frozen = True
    infer = Infer([(data, model)])

    # at the truth, stat ~ 0 and pseudo_residual ~ 0
    assert np.isclose(infer.calc_stat([2.0, 1.0]), 0.0, atol=1e-6)
    assert np.allclose(infer.calc_pseudo_residual([2.0, 1.0]), 0.0, atol=1e-3)
    assert np.isclose(infer.calc_loglike([2.0, 1.0]), -0.5 * infer.calc_stat([2.0, 1.0]))
    # clean labels strip LaTeX markup
    assert all('$' not in pl and '\\' not in pl for pl in infer.clean_free_plabels)
    assert len(infer.clean_free_indexed_plabels) == infer.free_nparams
