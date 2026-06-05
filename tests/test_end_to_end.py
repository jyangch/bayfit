import matplotlib

matplotlib.use('Agg')

import numpy as np

from curvefit.data.data import Data, DataUnit
from curvefit.infer.infer import BayesInfer
from curvefit.model.local import line
from curvefit.util.plot import Plot


def test_emcee_recovers_linear(tmp_path):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 30)
    ytrue = 2.0 * x + 1.0
    yerr = np.full(x.size, 0.5)
    y = ytrue + rng.normal(0, 0.5, x.size)

    unit = DataUnit(x, y, yerr=yerr, stat='chi2')
    data = Data([('d', unit)])
    model = line()
    model.params['logv'].frozen = True

    infer = BayesInfer([(data, model)])
    post = infer.emcee(nstep=400, discard=100, resume=False, savepath=str(tmp_path))

    k = post.par_best_ci[0]
    b = post.par_best_ci[1]
    assert abs(k - 2.0) < 0.3
    assert abs(b - 1.0) < 1.0

    fig = Plot.infer(post)
    assert fig is not None
    cfig = Plot.post_corner(post)
    assert cfig is not None


def test_plot_infer_handles_limit_points(tmp_path):
    # a dataset mixing detections, an upper limit, and a lower limit must
    # fit and plot without error
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 30)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, x.size)
    yerr = np.full(x.size, 0.5)

    up = np.zeros(x.size, dtype=bool)
    lo = np.zeros(x.size, dtype=bool)
    up[5] = True  # upper limit sits well above the line -> consistent
    y[5] = 2.0 * x[5] + 1.0 + 10.0
    lo[20] = True  # lower limit sits well below the line -> consistent
    y[20] = 2.0 * x[20] + 1.0 - 10.0

    unit = DataUnit(x, y, yerr=yerr, ups=up, los=lo, stat='chi2')
    data = Data([('d', unit)])
    model = line()
    model.params['logv'].frozen = True
    # start from a feasible point where both limits are already satisfied
    model.params['k'].val = 2.0
    model.params['b'].val = 1.0

    infer = BayesInfer([(data, model)])
    post = infer.emcee(nstep=400, discard=100, resume=False, savepath=str(tmp_path))

    assert abs(post.par_best_ci[0] - 2.0) < 0.3

    fig = Plot.infer(post)
    assert fig is not None
