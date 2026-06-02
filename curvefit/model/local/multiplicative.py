"""Multiplicative curve components for generic x/y curve fitting.

Each class returns a dimensionless factor that multiplies an additive
component, letting the user apply energy-independent or x-dependent
suppression to a base model.
"""

from collections import OrderedDict

import numpy as np

from ...util.param import Par
from ...util.prior import unif
from ..model import Multiplicative


class expcut(Multiplicative):
    r"""Exponential cutoff factor :math:`e^{-x/x_c}` applied to a base model."""

    def __init__(self):
        r"""Initialise the exponential cutoff with log-cutoff parameter ``logxc``."""
        super().__init__()

        self.expr = 'expcut'
        self.comment = 'exponential cutoff model'

        self.params = OrderedDict()
        self.params['logxc'] = Par(0, unif(-10, 10))

    def func(self, X):
        r"""Return the multiplicative factor :math:`e^{-x/x_c}` on ``X[:, 0]``.

        Args:
            X: 2-D input array; the x-grid is taken from column 0.

        Returns:
            1-D array of dimensionless factors in ``(0, 1]`` with the same
            length as ``X``.
        """
        x = X[:, 0]

        xc = 10 ** self.params['logxc'].value

        return np.exp(-x / xc)
