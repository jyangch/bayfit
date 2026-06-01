import numpy as np

from curvefit.data.data import Data, DataUnit
from curvefit.infer.pair import Pair
from curvefit.model.local import ln


def test_loglike_on_exact_linear_data():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    yerr = np.full(4, 0.1)
    unit = DataUnit(x, y, yerr=yerr, stat='chi^2')
    data = Data([('d', unit)])

    model = ln()
    model.params['k'].val = 2.0
    model.params['b'].val = 1.0
    model.params['logv'].frozen = True  # unused by chi^2

    pair = Pair(data, model)
    # model matches data exactly -> chi^2 ~ 0 -> loglike ~ 0
    assert np.isclose(pair.loglike, 0.0)
    assert np.isclose(pair.stat, 0.0)
    assert pair.npoint == 4


def test_loglike_penalizes_bad_fit():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    unit = DataUnit(x, y, yerr=np.full(4, 0.1), stat='chi^2')
    data = Data([('d', unit)])

    model = ln()
    model.params['k'].val = 0.0
    model.params['b'].val = 0.0
    pair = Pair(data, model)
    assert pair.loglike < 0.0
