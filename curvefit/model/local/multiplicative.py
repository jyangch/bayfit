from collections import OrderedDict

import numpy as np

from ...util.param import Par
from ...util.prior import unif
from ..model import Multiplicative


class expcut(Multiplicative):
    def __init__(self):
        super().__init__()

        self.expr = 'expcut'
        self.comment = 'exponential cutoff model'

        self.params = OrderedDict()
        self.params['logxc'] = Par(0, unif(-10, 10))

    def func(self, X):
        x = X[:, 0]

        xc = 10 ** self.params['logxc'].value

        return np.exp(-x / xc)
