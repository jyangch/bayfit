from collections import OrderedDict

import numpy as np
from scipy.integrate import solve_ivp

from ...util.param import Par
from ...util.prior import unif
from ..model import Additive


class ln(Additive):
    def __init__(self):
        super().__init__()

        self.expr = 'ln'
        self.comment = 'linear model'

        self.params = OrderedDict()
        self.params['k'] = Par(0, unif(-10, 10))
        self.params['b'] = Par(0, unif(-10, 10))
        self.params['logv'] = Par(0, unif(-10, 10))

    def func(self, X):

        x = X[:, 0]

        k = self.params['k'].value
        b = self.params['b'].value

        return k * x + b


class pl(Additive):
    def __init__(self):
        super().__init__()

        self.expr = 'pl'
        self.comment = 'power law model'

        self.params = OrderedDict()
        self.params['alpha'] = Par(0, unif(-10, 10))
        self.params['logA'] = Par(0, unif(-10, 10))

    def func(self, X):

        x = X[:, 0]

        alpha = self.params['alpha'].value
        logA = self.params['logA'].value

        return 10**logA * x**alpha


class expd(Additive):
    def __init__(self):
        super().__init__()

        self.expr = 'expd'
        self.comment = 'exponential decay model'

        self.params = OrderedDict()
        self.params['tau'] = Par(5, unif(0, 10))
        self.params['logA'] = Par(0, unif(-10, 10))

    def func(self, X):

        x = X[:, 0]

        tau = self.params['tau'].value
        logA = self.params['logA'].value

        return 10**logA * np.exp(-x / tau)


class bln(Additive):
    def __init__(self, seg):
        super().__init__()

        if not isinstance(seg, int):
            raise ValueError('expected type for seg is int')

        self.seg = seg
        self.expr = f'{seg}-bln'
        self.comment = 'broken lines'

        self.params = OrderedDict()

        if seg == 1:
            self.params['k1'] = Par(0, unif(-10, 10))
            self.params['b'] = Par(0, unif(-10, 10))

        elif seg == 2:
            self.params['k1'] = Par(0, unif(-10, 10))
            self.params['k2'] = Par(0, unif(-10, 10))
            self.params['x1'] = Par(0, unif(-10, 10))
            self.params['b'] = Par(0, unif(-10, 10))

        elif seg == 3:
            self.params['k1'] = Par(0, unif(-10, 10))
            self.params['k2'] = Par(0, unif(-10, 10))
            self.params['k3'] = Par(0, unif(-10, 10))
            self.params['x1'] = Par(0, unif(-10, 10))
            self.params['x2'] = Par(0, unif(-10, 10))
            self.params['b'] = Par(0, unif(-10, 10))

        elif seg == 4:
            self.params['k1'] = Par(0, unif(-10, 10))
            self.params['k2'] = Par(0, unif(-10, 10))
            self.params['k3'] = Par(0, unif(-10, 10))
            self.params['k4'] = Par(0, unif(-10, 10))
            self.params['x1'] = Par(0, unif(-10, 10))
            self.params['x2'] = Par(0, unif(-10, 10))
            self.params['x3'] = Par(0, unif(-10, 10))
            self.params['b'] = Par(0, unif(-10, 10))

        else:
            raise ValueError('unsupported seg value')

    def func(self, X):

        x = X[:, 0]
        theta = [par.value for par in self.params.values()]

        if self.seg == 1:
            return self.b1ln(x, theta)

        elif self.seg == 2:
            return self.b2ln(x, theta)

        elif self.seg == 3:
            return self.b3ln(x, theta)

        elif self.seg == 4:
            return self.b4ln(x, theta)

        else:
            raise ValueError('unsupported seg value')

    @staticmethod
    def b1ln(x, theta):

        x = np.array(x)
        k1, b = theta

        def l1(x):
            return k1 * x + b

        return l1(x)

    @staticmethod
    def b2ln(x, theta):

        x = np.array(x)
        k1, k2, x1, b = theta

        def l1(x):
            return k1 * x + b

        def l2(x):
            return k2 * (x - x1) + l1(x1)

        return l1(x) * (x <= x1) + l2(x) * (x > x1)

    @staticmethod
    def b3ln(x, theta):

        x = np.array(x)
        k1, k2, k3, x1, x2, b = theta

        def l1(x):
            return k1 * x + b

        def l2(x):
            return k2 * (x - x1) + l1(x1)

        def l3(x):
            return k3 * (x - x2) + l2(x2)

        return l1(x) * (x <= x1) + l2(x) * ((x > x1) & (x <= x2)) + l3(x) * (x > x2)

    @staticmethod
    def b4ln(x, theta):

        x = np.array(x)
        k1, k2, k3, k4, x1, x2, x3, b = theta

        def l1(x):
            return k1 * x + b

        def l2(x):
            return k2 * (x - x1) + l1(x1)

        def l3(x):
            return k3 * (x - x2) + l2(x2)

        def l4(x):
            return k4 * (x - x3) + l3(x3)

        return (
            l1(x) * (x <= x1)
            + l2(x) * ((x > x1) & (x <= x2))
            + l3(x) * ((x > x2) & (x <= x3))
            + l4(x) * (x > x3)
        )


class bpl(Additive):
    def __init__(self, seg):
        super().__init__()

        if not isinstance(seg, int):
            raise ValueError('expected type for seg is int')

        self.seg = seg
        self.expr = f'{seg}-bpl'
        self.comment = 'broken power laws'

        self.params = OrderedDict()

        if seg == 1:
            self.params['k1'] = Par(0, unif(-10, 10))
            self.params['logA'] = Par(0, unif(-10, 10))

        elif seg == 2:
            self.params['k1'] = Par(0, unif(-10, 10))
            self.params['k2'] = Par(0, unif(-10, 10))
            self.params['x1'] = Par(0, unif(-10, 10))
            self.params['logA'] = Par(0, unif(-10, 10))

        elif seg == 3:
            self.params['k1'] = Par(0, unif(-10, 10))
            self.params['k2'] = Par(0, unif(-10, 10))
            self.params['k3'] = Par(0, unif(-10, 10))
            self.params['x1'] = Par(0, unif(-10, 10))
            self.params['x2'] = Par(0, unif(-10, 10))
            self.params['logA'] = Par(0, unif(-10, 10))

        elif seg == 4:
            self.params['k1'] = Par(0, unif(-10, 10))
            self.params['k2'] = Par(0, unif(-10, 10))
            self.params['k3'] = Par(0, unif(-10, 10))
            self.params['k4'] = Par(0, unif(-10, 10))
            self.params['x1'] = Par(0, unif(-10, 10))
            self.params['x2'] = Par(0, unif(-10, 10))
            self.params['x3'] = Par(0, unif(-10, 10))
            self.params['logA'] = Par(0, unif(-10, 10))

        else:
            raise ValueError('unsupported seg value')

    def func(self, X):

        x = X[:, 0]
        theta = [par.value for par in self.params.values()]

        if self.seg == 1:
            return self.b1pl(x, theta)

        elif self.seg == 2:
            return self.b2pl(x, theta)

        elif self.seg == 3:
            return self.b3pl(x, theta)

        elif self.seg == 4:
            return self.b4pl(x, theta)

        else:
            raise ValueError('unsupported seg value')

    @staticmethod
    def b1pl(x, theta):

        x = np.array(x)
        k1, logA = theta

        def pl1(x):
            return 10**logA * x**k1

        return pl1(x)

    @staticmethod
    def b2pl(x, theta):

        x = np.array(x)
        k1, k2, x1, logA = theta

        def pl1(x):
            return 10**logA * x**k1

        def pl2(x):
            return pl1(x1) * (x / x1) ** k2

        return pl1(x) * (x <= x1) + pl2(x) * (x > x1)

    @staticmethod
    def b3pl(x, theta):

        x = np.array(x)
        k1, k2, k3, x1, x2, logA = theta

        def pl1(x):
            return 10**logA * x**k1

        def pl2(x):
            return pl1(x1) * (x / x1) ** k2

        def pl3(x):
            return pl2(x2) * (x / x2) ** k3

        return pl1(x) * (x <= x1) + pl2(x) * ((x > x1) & (x <= x2)) + pl3(x) * (x > x2)

    @staticmethod
    def b4pl(x, theta):

        x = np.array(x)
        k1, k2, k3, k4, x1, x2, x3, logA = theta

        def pl1(x):
            return 10**logA * x**k1

        def pl2(x):
            return pl1(x1) * (x / x1) ** k2

        def pl3(x):
            return pl2(x2) * (x / x2) ** k3

        def pl4(x):
            return pl3(x3) * (x / x3) ** k4

        return (
            pl1(x) * (x <= x1)
            + pl2(x) * ((x > x1) & (x <= x2))
            + pl3(x) * ((x > x2) & (x <= x3))
            + pl4(x) * (x > x3)
        )


class sbpl(Additive):
    def __init__(self, seg):
        super().__init__()

        if not isinstance(seg, int):
            raise ValueError('expected type for seg is int')

        self.seg = seg
        self.expr = f'{seg}-sbpl'
        self.comment = 'smoothly broken power laws'

        self.params = OrderedDict()

        if seg == 2:
            self.params['k1'] = Par(0, unif(-10, 10))
            self.params['k2'] = Par(0, unif(-10, 10))
            self.params['x1'] = Par(0, unif(-10, 10))
            self.params['logO1'] = Par(0, unif(-1, 1))
            self.params['logA'] = Par(0, unif(-10, 10))

        elif seg == 3:
            self.params['k1'] = Par(0, unif(-10, 10))
            self.params['k2'] = Par(0, unif(-10, 10))
            self.params['k3'] = Par(0, unif(-10, 10))
            self.params['x1'] = Par(0, unif(-10, 10))
            self.params['x2'] = Par(0, unif(-10, 10))
            self.params['logO1'] = Par(0, unif(-1, 1))
            self.params['logO2'] = Par(0, unif(-1, 1))
            self.params['logA'] = Par(0, unif(-10, 10))

        elif seg == 4:
            self.params['k1'] = Par(0, unif(-10, 10))
            self.params['k2'] = Par(0, unif(-10, 10))
            self.params['k3'] = Par(0, unif(-10, 10))
            self.params['k4'] = Par(0, unif(-10, 10))
            self.params['x1'] = Par(0, unif(-10, 10))
            self.params['x2'] = Par(0, unif(-10, 10))
            self.params['x3'] = Par(0, unif(-10, 10))
            self.params['logO1'] = Par(0, unif(-1, 1))
            self.params['logO2'] = Par(0, unif(-1, 1))
            self.params['logO3'] = Par(0, unif(-1, 1))
            self.params['logA'] = Par(0, unif(-10, 10))

        else:
            raise ValueError('unsupported seg value')

    def func(self, X):

        x = X[:, 0]
        theta = [par.value for par in self.params.values()]

        if self.seg == 2:
            return self.sb2pl(x, theta)

        elif self.seg == 3:
            return self.sb3pl(x, theta)

        elif self.seg == 4:
            return self.sb4pl(x, theta)

        else:
            raise ValueError('unsupported seg value')

    @staticmethod
    def pl(x, theta):

        k1, x1, logA = theta
        Ampl = 10**logA

        return Ampl * (x / x1) ** (k1)

    @staticmethod
    def sb2pl(x, theta):

        k1, k2, x1, logO1, logA = theta
        omega1 = 10**logO1

        F1 = sbpl.pl(x, [k1, x1, logA])
        F2 = sbpl.pl(x, [k2, x1, logA])
        F12 = (F1 ** (-omega1) + F2 ** (-omega1)) ** (-1 / omega1)

        return F12

    @staticmethod
    def sb3pl(x, theta):

        k1, k2, k3, x1, x2, logO1, logO2, logA = theta
        omega2 = 10**logO2

        F12 = sbpl.sb2pl(x, [k1, k2, x1, logO1, logA])
        F3 = sbpl.pl(x, [k3, x2, np.log10(sbpl.sb2pl(x2, [k1, k2, x1, logO1, logA]))])
        F123 = (F12 ** (-omega2) + F3 ** (-omega2)) ** (-1 / omega2)

        return F123

    @staticmethod
    def sb4pl(x, theta):

        k1, k2, k3, k4, x1, x2, x3, logO1, logO2, logO3, logA = theta
        omega3 = 10**logO3

        F123 = sbpl.sb3pl(x, [k1, k2, k3, x1, x2, logO1, logO2, logA])
        F4 = sbpl.pl(
            x, [k4, x3, np.log10(sbpl.sb3pl(x3, [k1, k2, k3, x1, x2, logO1, logO2, logA]))]
        )
        F1234 = (F123 ** (-omega3) + F4 ** (-omega3)) ** (-1 / omega3)

        return F1234


class spindown(Additive):
    def __init__(self):
        super().__init__()

        self.expr = 'spindown'
        self.comment = 'magnetar spin-down model'

        self.params = OrderedDict()
        self.params['log$B_{p,15}$'] = Par(0, unif(-1, 1))
        self.params['log$P_{0,-3}$'] = Par(0, unif(-1, 1))
        self.params['log$\\eta_{-3}$'] = Par(0, unif(-1, 1))

    def ode(self, t, y, a, b):
        return a * y**3 + b * y**5

    def func(self, X):

        t = X[:, 0]

        Bp = 10 ** self.params['log$B_{p,15}$'].value * 1e15
        P0 = 10 ** self.params['log$P_{0,-3}$'].value * 1e-3
        eta = 10 ** self.params['log$\\eta_{-3}$'].value * 1e-3

        R_ = 1.2e6
        c_ = 2.998e10
        G_ = 6.674e-8
        I_ = 3.33e45
        eps = 1.0e-4
        fb = 0.1

        T = 1e2
        Y = 1e3

        a = -(Bp**2) * R_**6 / (6 * c_**3 * I_)
        a_hat = a * T * Y**2

        b = -32 * G_ * I_ * eps**2 / (5 * c_**5)
        b_hat = b * T * Y**4

        t0 = 0
        y0 = 2.0 * np.pi / P0
        y0_hat = y0 / Y

        t_hat = t / T
        span = (t0, max(t_hat))

        y_hat = solve_ivp(self.ode, span, [y0_hat], args=(a_hat, b_hat), t_eval=t_hat)
        if y_hat.success:
            Omega = y_hat.y[0] * Y
        else:
            raise RuntimeError('ODE solver failed: ' + y_hat.message)

        return eta / fb * Bp**2 * R_**6 * Omega**4 / (6 * c_**3)


class psd(Additive):
    def __init__(self, expr):
        super().__init__()

        self.expr = expr
        self.comment = 'power spectral distribution model'

        self.params = OrderedDict()

        if self.expr == 'white':
            self.params['$A_w$'] = Par(2, unif(0, 5))

        elif self.expr == 'psd1':
            self.params['$A_w$'] = Par(2, unif(0, 5))
            self.params['$A_1$'] = Par(10, unif(0, 30))
            self.params['$\\nu_1$'] = Par(1000, unif(500, 5000))
            self.params['log$\\Delta\\nu_1$'] = Par(2, unif(1, 3))

        elif self.expr == 'psd2':
            self.params['$A_w$'] = Par(2, unif(0, 5))
            self.params['$A_1$'] = Par(10, unif(0, 30))
            self.params['$\\nu_1$'] = Par(1000, unif(500, 5000))
            self.params['log$\\Delta\\nu_1$'] = Par(2, unif(1, 3))
            self.params['$A_2$'] = Par(10, unif(0, 30))
            self.params['$\\nu_2$'] = Par(1000, unif(500, 5000))
            self.params['log$\\Delta\\nu_2$'] = Par(2, unif(1, 3))

        else:
            raise ValueError('unsupported expr value')

    def func(self, X):

        nu = X[:, 0]

        theta = [par.value for par in self.params.values()]

        if self.expr == 'white':
            return self.white(nu, theta)

        elif self.expr == 'psd1':
            return self.psd1(nu, theta)

        elif self.expr == 'psd2':
            return self.psd2(nu, theta)

        else:
            raise ValueError('unsupported expr value')

    @staticmethod
    def white(nu, cube):

        Aw = cube[0]
        Pnu = np.ones_like(nu) * Aw

        return Pnu

    @staticmethod
    def psd1(nu, cube):

        Aw = cube[0]
        A1 = cube[1]
        nu1 = cube[2]
        dnu1 = 10 ** cube[3]
        Pnu = Aw + A1 / (1 + (nu - nu1) ** 2 / dnu1**2)

        return Pnu

    @staticmethod
    def psd2(nu, cube):

        Aw = cube[0]
        A1 = cube[1]
        nu1 = cube[2]
        dnu1 = 10 ** cube[3]
        A2 = cube[4]
        nu2 = cube[5]
        dnu2 = 10 ** cube[6]
        Pnu = Aw + A1 / (1 + (nu - nu1) ** 2 / dnu1**2) + A2 / (1 + (nu - nu2) ** 2 / dnu2**2)

        return Pnu
