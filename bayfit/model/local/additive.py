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


class line(Additive):
    """Linear curve :math:`y = k x + b`."""

    def __init__(self):
        """Initialise linear model with slope ``k``, intercept ``b``, log-variance ``logv``."""

        self.expr = 'line'
        self.comment = 'linear model'

        self.config = OrderedDict()

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
        k = self.params['k'].value
        b = self.params['b'].value

        x, scalar = self._asx(X)

        y = k * x + b

        return y[0] if scalar else y


class pl(Additive):
    r"""Power-law curve :math:`y = 10^{\mathrm{logA}} \, x^{\alpha}`."""

    def __init__(self):
        """Initialise power-law model with index ``alpha`` and log-amplitude ``logA``."""

        self.expr = 'pl'
        self.comment = 'power law model'

        self.config = OrderedDict()

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
        alpha = self.params['alpha'].value
        logA = self.params['logA'].value

        amp = 10**logA

        x, scalar = self._asx(X)

        y = amp * x**alpha

        return y[0] if scalar else y


class expd(Additive):
    r"""Exponential decay curve :math:`y = 10^{\mathrm{logA}} \, e^{-x/\tau}`."""

    def __init__(self):
        """Initialise exponential decay model with timescale ``tau`` and log-amplitude ``logA``."""

        self.expr = 'expd'
        self.comment = 'exponential decay model'

        self.config = OrderedDict()

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
        tau = self.params['tau'].value
        logA = self.params['logA'].value

        amp = 10**logA

        x, scalar = self._asx(X)

        y = amp * np.exp(-x / tau)

        return y[0] if scalar else y


class spindown(Additive):
    """Magnetar spin-down luminosity model solved via an ODE for the spin frequency."""

    def __init__(self):
        """Initialise spin-down model with log-B, log-P0, and log-efficiency parameters."""
        super().__init__()

        self.expr = 'spindown'
        self.comment = 'magnetar spin-down model'

        self.params = OrderedDict()
        self.params[r'log$B_{p,15}$'] = Par(0, unif(-1, 1))
        self.params[r'log$P_{0,-3}$'] = Par(0, unif(-1, 1))
        self.params[r'log$\eta_{-3}$'] = Par(0, unif(-1, 1))

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
        t, scalar = self._asx(X)

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

        L = eta / fb * Bp**2 * R_**6 * Omega**4 / (6 * c_**3)

        return L[0] if scalar else L


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
        nu, scalar = self._asx(X)

        theta = [par.value for par in self.params.values()]

        if self.expr == 'white':
            y = self.white(nu, theta)

        elif self.expr == 'psd1':
            y = self.psd1(nu, theta)

        elif self.expr == 'psd2':
            y = self.psd2(nu, theta)

        else:
            raise ValueError('unsupported expr value')

        return y[0] if scalar else y

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
