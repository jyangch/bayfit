import matplotlib

matplotlib.use('Agg')

import numpy as np

from curvefit.model.local import ln
from curvefit.util.plot import Plot


def test_plot_model_returns_figure():
    m = ln()
    m.params['k'].val = 1.0
    m.params['b'].val = 0.0
    fig = Plot.model(m, np.linspace(0, 10, 50))
    assert fig is not None
    assert len(fig.axes) >= 1
