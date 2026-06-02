"""Mathematical modifier components for generic curve fitting.

Each class returns a value that is independent of the curve family (additive
or multiplicative) and acts as a dimensionless constant or scaling factor.
"""

from collections import OrderedDict

import numpy as np

from ...util.param import Par
from ...util.prior import unif
from ..model import Mathematic


class const(Mathematic):
    """Constant model: returns a uniform scalar value ``C`` at every x point."""

    def __init__(self):
        """Initialise the constant with a single free parameter ``C``."""
        super().__init__()

        self.expr = 'const'
        self.comment = 'constant model'

        self.params = OrderedDict()
        self.params['C'] = Par(1, unif(-10, 10))

    def func(self, X):
        """Return the constant value ``C`` broadcast over the x-grid in ``X[:, 0]``.

        Args:
            X: 2-D input array; the x-grid is taken from column 0 (used only
                to determine the output length; the values are not read).

        Returns:
            1-D array of length ``len(X)`` with every element equal to ``C``.
        """
        x = X[:, 0]

        C = self.params['C'].value

        return C * np.ones_like(x)
