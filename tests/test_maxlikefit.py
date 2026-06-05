import numpy as np
import pytest

from curvefit.data.data import Data, DataUnit
from curvefit.infer.analyzer import Bootstrap
from curvefit.infer.infer import MaxLikeFit
from curvefit.model.local import line


def _data(seed=0, stat='chi2'):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 40)
    yerr = np.full(x.size, 0.5)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, x.size)
    unit = DataUnit(x, y, yerr=yerr, stat=stat)
    return Data([('d', unit)])


def test_lmfit_recovers_linear():
    data = _data()
    model = line()
    model.params['logv'].frozen = True
    fit = MaxLikeFit([(data, model)])
    boot = fit.lmfit()
    assert isinstance(boot, Bootstrap)
    assert boot.bootstrap_sample.shape[1] == fit.free_nparams + 1
    k, b = boot.par_best[0], boot.par_best[1]
    assert abs(k - 2.0) < 0.3
    assert abs(b - 1.0) < 1.0


def test_iminuit_recovers_linear():
    data = _data(seed=1)
    model = line()
    model.params['logv'].frozen = True
    fit = MaxLikeFit([(data, model)])
    boot = fit.iminuit()
    assert isinstance(boot, Bootstrap)
    assert abs(boot.par_best[0] - 2.0) < 0.3


def test_iminuit_handles_chi2f_freevariance():
    data = _data(seed=2, stat='chi2f')
    model = line()  # k, b, logv all free
    fit = MaxLikeFit([(data, model)])
    boot = fit.iminuit()
    assert isinstance(boot, Bootstrap)
    assert abs(boot.par_best[0] - 2.0) < 0.5


def test_lmfit_rejects_free_variance_stats():
    data = _data(seed=3, stat='vdr')
    model = line()
    fit = MaxLikeFit([(data, model)])
    with pytest.raises(ValueError):
        fit.lmfit()
