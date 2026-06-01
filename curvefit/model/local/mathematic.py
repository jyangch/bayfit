from collections import OrderedDict

import numpy as np

from ...util.param import Par
from ...util.prior import unif
from ..model import Mathematic


class const(Mathematic):
    def __init__(self):
        super().__init__()

        self.expr = 'const'
        self.comment = 'constant model'

        self.params = OrderedDict()
        self.params['C'] = Par(1, unif(-10, 10))

    def func(self, X):
        x = X[:, 0]

        C = self.params['C'].value

        return C * np.ones_like(x)
