"""Additive curve components for generic x/y curve fitting.

Each class implements a concrete analytic family (linear, power-law,
exponential decay, broken lines, broken power-laws, smoothly broken
power-laws, magnetar spin-down, power spectral density) and returns the
model value from :meth:`func`. The input ``X`` is a 2-D array whose
first column ``X[:, 0]`` carries the x-grid.
"""

from collections import OrderedDict

import numpy as np
from scipy.integrate import solve_ivp

from ...util.param import Par
from ...util.prior import unif
from ..model import Additive


class ln(Additive):
    """Linear curve :math:`y = k x + b`."""

    def __init__(self):
        """Initialise linear model with slope ``k``, intercept ``b``, log-variance ``logv``."""
        super().__init__()

        self.expr = 'ln'
        self.comment = 'linear model'

        self.params = OrderedDict()
        self.params['k'] = Par(0, unif(-10, 10))
        self.params['b'] = Par(0, unif(-10, 10))
        self.params['logv'] = Par(0, unif(-10, 10))

    def func(self, X):
        r"""Return the linear model :math:`k x + b` evaluated on ``X[:, 0]``.

        Args:
            X: 2-D input array; the x-grid is taken from column 0.

        Returns:
            1-D array of model values with the same length as ``X``.
        """
        x = X[:, 0]

        k = self.params['k'].value
        b = self.params['b'].value

        return k * x + b


class pl(Additive):
    r"""Power-law curve :math:`y = 10^{\mathrm{logA}} \, x^{\alpha}`."""

    def __init__(self):
        """Initialise power-law model with index ``alpha`` and log-amplitude ``logA``."""
        super().__init__()

        self.expr = 'pl'
        self.comment = 'power law model'

        self.params = OrderedDict()
        self.params['alpha'] = Par(0, unif(-10, 10))
        self.params['logA'] = Par(0, unif(-10, 10))

    def func(self, X):
        r"""Return the power-law :math:`10^{\mathrm{logA}} \, x^{\alpha}` on ``X[:, 0]``.

        Args:
            X: 2-D input array; the x-grid is taken from column 0.

        Returns:
            1-D array of model values with the same length as ``X``.
        """
        x = X[:, 0]

        alpha = self.params['alpha'].value
        logA = self.params['logA'].value

        return 10**logA * x**alpha


class expd(Additive):
    r"""Exponential decay curve :math:`y = 10^{\mathrm{logA}} \, e^{-x/\tau}`."""

    def __init__(self):
        """Initialise exponential decay model with timescale ``tau`` and log-amplitude ``logA``."""
        super().__init__()

        self.expr = 'expd'
        self.comment = 'exponential decay model'

        self.params = OrderedDict()
        self.params['tau'] = Par(5, unif(0, 10))
        self.params['logA'] = Par(0, unif(-10, 10))

    def func(self, X):
        r"""Return the exponential decay :math:`10^{\mathrm{logA}} e^{-x/\tau}` on ``X[:, 0]``.

        Args:
            X: 2-D input array; the x-grid is taken from column 0.

        Returns:
            1-D array of model values with the same length as ``X``.
        """
        x = X[:, 0]

        tau = self.params['tau'].value
        logA = self.params['logA'].value

        return 10**logA * np.exp(-x / tau)


class bln(Additive):
    """Broken lines: piecewise-linear curve with 1-4 segments joined at break points."""

    def __init__(self, seg):
        """Initialise the broken-line model for the requested number of segments.

        Args:
            seg: Number of linear segments; must be an integer in ``{1, 2, 3, 4}``.

        Raises:
            ValueError: If ``seg`` is not an integer, or is outside ``{1, 2, 3, 4}``.
        """
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
        """Return the broken-line curve dispatched to the appropriate segment helper.

        Args:
            X: 2-D input array; the x-grid is taken from column 0.

        Returns:
            1-D array of model values with the same length as ``X``.

        Raises:
            ValueError: If ``self.seg`` is outside the supported range ``{1, 2, 3, 4}``.
        """
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
        """Return a single-segment line ``k1 * x + b``.

        The ``b1ln``, ``b2ln``, ``b3ln``, and ``b4ln`` family evaluate the
        corresponding 1-4 segment broken-line model on raw arrays.  Each
        segment is continuous: the value at each break point is inherited by
        the next segment so the curve has no discontinuities.

        Args:
            x: 1-D x-grid array.
            theta: Flat list of parameter values in the order used by the
                parent ``bln`` ``params`` dict for the matching ``seg``.

        Returns:
            1-D array of the same shape as ``x``.
        """
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
    """Broken power-laws: piecewise power-law curve with 1-4 segments joined at break points."""

    def __init__(self, seg):
        """Initialise the broken power-law model for the requested number of segments.

        Args:
            seg: Number of power-law segments; must be an integer in ``{1, 2, 3, 4}``.

        Raises:
            ValueError: If ``seg`` is not an integer, or is outside ``{1, 2, 3, 4}``.
        """
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
        """Return the broken power-law curve dispatched to the appropriate segment helper.

        Args:
            X: 2-D input array; the x-grid is taken from column 0.

        Returns:
            1-D array of model values with the same length as ``X``.

        Raises:
            ValueError: If ``self.seg`` is outside the supported range ``{1, 2, 3, 4}``.
        """
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
        """Return a single-segment power-law :math:`10^{\\mathrm{logA}} x^{k_1}`.

        The ``b1pl``, ``b2pl``, ``b3pl``, and ``b4pl`` family evaluate the
        corresponding 1-4 segment broken power-law model on raw arrays.  Each
        segment is continuous: the amplitude at each break is inherited by the
        next segment so the curve has no discontinuities.

        Args:
            x: 1-D x-grid array.
            theta: Flat list of parameter values in the order used by the
                parent ``bpl`` ``params`` dict for the matching ``seg``.

        Returns:
            1-D array of the same shape as ``x``.
        """
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
    """Smoothly broken power-laws: power-law segments joined by smooth transitions."""

    def __init__(self, seg):
        """Initialise the smoothly broken power-law for the requested number of segments.

        The smoothness of each break is controlled by a log-omega parameter
        (``logO1``, ``logO2``, ``logO3``); larger omega gives a sharper transition.

        Args:
            seg: Number of power-law segments; must be an integer in ``{2, 3, 4}``.

        Raises:
            ValueError: If ``seg`` is not an integer, or is outside ``{2, 3, 4}``.
        """
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
        """Return the smoothly broken power-law dispatched to the appropriate helper.

        Args:
            X: 2-D input array; the x-grid is taken from column 0.

        Returns:
            1-D array of model values with the same length as ``X``.

        Raises:
            ValueError: If ``self.seg`` is outside the supported range ``{2, 3, 4}``.
        """
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
        r"""Return a single unnormalised power-law :math:`10^{\mathrm{logA}} (x/x_1)^{k_1}`.

        Args:
            x: 1-D x-grid array.
            theta: List ``[k1, x1, logA]`` -- index, pivot, log-amplitude.

        Returns:
            1-D array of the same shape as ``x``.
        """
        k1, x1, logA = theta
        Ampl = 10**logA

        return Ampl * (x / x1) ** (k1)

    @staticmethod
    def sb2pl(x, theta):
        r"""Return the 2-segment smoothly broken power-law.

        The ``sb2pl``, ``sb3pl``, and ``sb4pl`` family build up the
        multi-segment smoothly broken power-law recursively. Each break is
        smoothed via the joining formula
        :math:`(F_1^{-\omega} + F_2^{-\omega})^{-1/\omega}`, where
        :math:`\omega = 10^{\mathrm{logO}}` controls sharpness.

        Args:
            x: 1-D x-grid array.
            theta: Flat list of parameter values in the order used by the
                parent ``sbpl`` ``params`` dict for ``seg == 2``.

        Returns:
            1-D array of the same shape as ``x``.
        """

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
    """Magnetar spin-down luminosity model solved via an ODE for the spin frequency."""

    def __init__(self):
        """Initialise spin-down model with log-B, log-P0, and log-efficiency parameters."""
        super().__init__()

        self.expr = 'spindown'
        self.comment = 'magnetar spin-down model'

        self.params = OrderedDict()
        self.params['log$B_{p,15}$'] = Par(0, unif(-1, 1))
        self.params['log$P_{0,-3}$'] = Par(0, unif(-1, 1))
        self.params['log$\\eta_{-3}$'] = Par(0, unif(-1, 1))

    def ode(self, t, y, a, b):
        r"""Return the spin-down ODE right-hand side :math:`a \Omega^3 + b \Omega^5`.

        Encodes magnetic dipole (``a``) and gravitational-wave (``b``) braking
        for the dimensionless spin-frequency ``y``.

        Args:
            t: Current time (dimensionless, scaled by ``T``).
            y: Current state vector ``[Omega_hat]`` (dimensionless spin frequency).
            a: Magnetic-dipole braking coefficient (dimensionless, pre-scaled).
            b: Gravitational-wave braking coefficient (dimensionless, pre-scaled).

        Returns:
            Array with one element: the time derivative of ``y``.
        """
        return a * y**3 + b * y**5

    def func(self, X):
        """Return the magnetar spin-down luminosity at each time in ``X[:, 0]``.

        Integrates the spin-frequency ODE numerically with :func:`scipy.integrate.solve_ivp`
        using the physical magnetar constants stored as local variables, then
        converts the resulting spin frequency to radiated power.

        Args:
            X: 2-D input array; the time grid (in seconds) is taken from column 0.

        Returns:
            1-D array of luminosity values (in erg/s) with the same length as ``X``.

        Raises:
            RuntimeError: If the ODE solver fails to converge.
        """
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
    """Power spectral density model: white noise plus up to two Lorentzian components."""

    def __init__(self, expr):
        """Initialise the PSD model for the requested variant.

        Args:
            expr: Model variant string; one of ``'white'``, ``'psd1'``, or
                ``'psd2'``. ``'white'`` is flat noise only; ``'psd1'`` adds
                one Lorentzian peak; ``'psd2'`` adds two Lorentzian peaks.

        Raises:
            ValueError: If ``expr`` is not one of the supported variant strings.
        """
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
        """Return the PSD model value at each frequency in ``X[:, 0]``.

        Args:
            X: 2-D input array; the frequency grid is taken from column 0.

        Returns:
            1-D array of PSD values with the same length as ``X``.

        Raises:
            ValueError: If ``self.expr`` is not a supported variant string.
        """
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
        """Return a flat (white) noise spectrum with constant power ``Aw``.

        The ``white``, ``psd1``, and ``psd2`` family evaluate the corresponding
        PSD variant on raw frequency arrays. ``psd1`` adds one Lorentzian peak
        and ``psd2`` adds two independent Lorentzian peaks to the white floor.

        Args:
            nu: 1-D frequency-grid array.
            cube: List of parameter values in the order used by the parent
                ``psd`` ``params`` dict for the matching ``expr`` variant.

        Returns:
            1-D array of PSD values with the same shape as ``nu``.
        """
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
