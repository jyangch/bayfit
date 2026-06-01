import matplotlib

matplotlib.use('Agg')

import numpy as np

from curvefit.data.data import Data, DataUnit
from curvefit.infer.infer import Infer
from curvefit.model.local import ln
from curvefit.util.plot import Plot


def test_emcee_recovers_linear(tmp_path):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 30)
    ytrue = 2.0 * x + 1.0
    yerr = np.full(x.size, 0.5)
    y = ytrue + rng.normal(0, 0.5, x.size)

    unit = DataUnit(x, y, yerr=yerr, stat='chi^2')
    data = Data([('d', unit)])
    model = ln()
    model.params['logv'].frozen = True

    infer = Infer([(data, model)])
    post = infer.emcee(nstep=400, discard=100, resume=False, savepath=str(tmp_path))

    k = post.par_best_ci[0]
    b = post.par_best_ci[1]
    assert abs(k - 2.0) < 0.3
    assert abs(b - 1.0) < 1.0

    fig = Plot.infer(post, nsample=50, ngrid=50)
    assert fig is not None
    cfig = Plot.post_corner(post)
    assert cfig is not None
