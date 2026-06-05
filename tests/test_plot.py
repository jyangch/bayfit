import matplotlib

matplotlib.use('Agg')

import numpy as np

from bayfit.model.local import line
from bayfit.util.plot import Plot


def test_plot_model_returns_figure():
    m = line()
    m.params['k'].val = 1.0
    m.params['b'].val = 0.0
    mp = Plot.model(ploter='matplotlib')
    mp.add_model(m, np.linspace(0, 10, 50)[:, None])
    fig = mp.get_fig()
    assert fig is not None
    assert len(fig.fig.axes) >= 1
