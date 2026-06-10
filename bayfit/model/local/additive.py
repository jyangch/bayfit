"""Additive curve components for generic x/y curve fitting.

Each class implements a concrete analytic family (linear, power-law,
exponential decay, single/double/triple smoothly broken power-laws,
magnetar spin-down, power spectral density) and returns the model value
from :meth:`func`. The input ``X`` is a 2-D array whose first column
``X[:, 0]`` carries the x-grid.
"""

from collections import OrderedDict

import numpy as np
from scipy.integrate import solve_ivp

from bayfit.util.tools import cached_property

from ...util.param import Cfg, Par
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
        self.params[r'$k$'] = Par(0, unif(-10, 10))
        self.params[r'$b$'] = Par(0, unif(-10, 10))
        self.params[r'log$v$'] = Par(0, unif(-10, 10))

    def func(self, X):
        r"""Return the linear model :math:`k x + b` evaluated on ``X[:, 0]``.

        Args:
            X: 2-D input array; the x-grid is taken from column 0.

        Returns:
            1-D array of model values with the same length as ``X``.
        """

        x, scalar = self.asx(X)

        k = self.params[r'$k$'].value
        b = self.params[r'$b$'].value

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
        self.params[r'$\alpha$'] = Par(0, unif(-10, 10))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))

    def func(self, X):
        r"""Return the power-law :math:`10^{\mathrm{logA}} \, x^{\alpha}` on ``X[:, 0]``.

        Args:
            X: 2-D input array; the x-grid is taken from column 0 and must be
                strictly positive.

        Returns:
            1-D array of model values with the same length as ``X``.

        Raises:
            ValueError: If any ``x`` is non-positive (non-integer powers of
                ``x <= 0`` are undefined).
        """

        x, scalar = self.asx(X)

        if np.any(x <= 0):
            raise ValueError('pl requires positive x (non-integer powers of x<=0 are undefined)')

        alpha = self.params[r'$\alpha$'].value
        amp = 10 ** self.params[r'log$A$'].value

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
        self.params[r'$\tau$'] = Par(5, unif(1e-3, 10))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))

    def func(self, X):
        r"""Return the exponential decay :math:`10^{\mathrm{logA}} e^{-x/\tau}` on ``X[:, 0]``.

        Args:
            X: 2-D input array; the x-grid is taken from column 0.

        Returns:
            1-D array of model values with the same length as ``X``.
        """

        x, scalar = self.asx(X)

        tau = self.params[r'$\tau$'].value
        amp = 10 ** self.params[r'log$A$'].value

        y = amp * np.exp(-x / tau)

        return y[0] if scalar else y


class sbpl(Additive):
    """Smoothly broken power law (Kaneko et al. 2006, ``10.1086/505911``)."""

    def __init__(self):
        """Initialise sbpl; params switch on ``vfv_peak`` config."""

        self.expr = 'sbpl'
        self.comment = 'smoothly broken power-law model'

        self.config = OrderedDict()
        self.config['pivot'] = Cfg(1.0)

        self.params = OrderedDict()
        self.params[r'$\alpha_1$'] = Par(0, unif(-10, 10))
        self.params[r'$\alpha_2$'] = Par(0, unif(-10, 10))
        self.params[r'log$x_b$'] = Par(2, unif(0, 4))
        self.params[r'$\delta$'] = Par(0.3, unif(1e-3, 3))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))

    @staticmethod
    def _log_cosh(q):
        r"""Return :math:`\ln\cosh q`, computed stably via :func:`numpy.logaddexp`.

        Equivalent to ``log(cosh(q))`` but avoids overflow for large ``|q|``.

        Args:
            q: Array of dimensionless log-scale distances from a break.

        Returns:
            Array of :math:`\ln\cosh q` with the same shape as ``q``.
        """

        return np.logaddexp(q, -q) - np.log(2.0)

    def func(self, X):
        r"""Return the smoothly broken power law evaluated on ``X[:, 0]``.

        Implements the Kaneko et al. (2006) SBPL: a power law of mean slope
        :math:`b = (\alpha_1 + \alpha_2)/2` whose local index transitions
        smoothly from :math:`\alpha_1` (low ``x``) to :math:`\alpha_2`
        (high ``x``) across a break at :math:`x_b` with smoothness
        :math:`\delta`. The curve is normalised to the amplitude :math:`A`
        at the ``pivot`` config value.

        Args:
            X: Scalar, 1-D, or 2-D input; the x-grid is taken from column 0
                and must be strictly positive.

        Returns:
            Model values matching the input sampling (scalar in, scalar out).

        Raises:
            ValueError: If any ``x`` is non-positive (the model lives on a
                log scale).
        """

        x, scalar = self.asx(X)

        if np.any(x <= 0):
            raise ValueError('sbpl requires positive x (evaluated on a log scale)')

        xpiv = self.config['pivot'].value

        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        xb = 10 ** self.params[r'log$x_b$'].value
        delta = self.params[r'$\delta$'].value
        amp = 10 ** self.params[r'log$A$'].value

        b = (alpha1 + alpha2) / 2
        m = (alpha2 - alpha1) / 2

        q = np.log10(x / xb) / delta
        qpiv = np.log10(xpiv / xb) / delta

        a = m * delta * self._log_cosh(q)
        apiv = m * delta * self._log_cosh(qpiv)

        y = amp * (x / xpiv) ** b * 10 ** (a - apiv)

        return y[0] if scalar else y


class dsbpl(Additive):
    """Double smoothly broken power law (three segments, two smooth breaks)."""

    def __init__(self):
        """Initialise double sbpl with three indices and two breaks."""

        self.expr = 'dsbpl'
        self.comment = 'double smoothly broken power-law model'

        self.config = OrderedDict()
        self.config['pivot'] = Cfg(1.0)

        self.params = OrderedDict()
        self.params[r'$\alpha_1$'] = Par(0, unif(-10, 10))
        self.params[r'$\alpha_2$'] = Par(0, unif(-10, 10))
        self.params[r'$\alpha_3$'] = Par(0, unif(-10, 10))
        self.params[r'log$x_{b1}$'] = Par(2, unif(0, 4))
        self.params[r'log$x_{b2}$'] = Par(2, unif(0, 4))
        self.params[r'$\delta_1$'] = Par(0.3, unif(1e-3, 3))
        self.params[r'$\delta_2$'] = Par(0.3, unif(1e-3, 3))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))

    @staticmethod
    def _log_cosh(q):
        r"""Return :math:`\ln\cosh q`, computed stably via :func:`numpy.logaddexp`.

        Equivalent to ``log(cosh(q))`` but avoids overflow for large ``|q|``.

        Args:
            q: Array of dimensionless log-scale distances from a break.

        Returns:
            Array of :math:`\ln\cosh q` with the same shape as ``q``.
        """

        return np.logaddexp(q, -q) - np.log(2.0)

    def func(self, X):
        r"""Return the double smoothly broken power law on ``X[:, 0]``.

        Three power-law segments joined by two smooth breaks
        (:math:`x_{b1}, x_{b2}`): the local index runs
        :math:`\alpha_1 \to \alpha_2 \to \alpha_3` with mean slope
        :math:`b = (\alpha_1 + \alpha_3)/2`, each break smoothed by its own
        :math:`\delta_i`. The curve is normalised to the amplitude
        :math:`A` at the ``pivot`` config value.

        Args:
            X: Scalar, 1-D, or 2-D input; the x-grid is taken from column 0
                and must be strictly positive.

        Returns:
            Model values matching the input sampling (scalar in, scalar out).

        Raises:
            ValueError: If any ``x`` is non-positive (the model lives on a
                log scale).
        """

        x, scalar = self.asx(X)

        if np.any(x <= 0):
            raise ValueError('dsbpl requires positive x (evaluated on a log scale)')

        xpiv = self.config['pivot'].value

        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        alpha3 = self.params[r'$\alpha_3$'].value
        xb1 = 10 ** self.params[r'log$x_{b1}$'].value
        xb2 = 10 ** self.params[r'log$x_{b2}$'].value
        delta1 = self.params[r'$\delta_1$'].value
        delta2 = self.params[r'$\delta_2$'].value
        amp = 10 ** self.params[r'log$A$'].value

        b = 0.5 * (alpha1 + alpha3)
        m1 = 0.5 * (alpha2 - alpha1)
        m2 = 0.5 * (alpha3 - alpha2)

        q1 = np.log10(x / xb1) / delta1
        q2 = np.log10(x / xb2) / delta2

        qpiv1 = np.log10(xpiv / xb1) / delta1
        qpiv2 = np.log10(xpiv / xb2) / delta2

        a1 = m1 * delta1 * self._log_cosh(q1)
        a2 = m2 * delta2 * self._log_cosh(q2)

        apiv1 = m1 * delta1 * self._log_cosh(qpiv1)
        apiv2 = m2 * delta2 * self._log_cosh(qpiv2)

        y = amp * (x / xpiv) ** b * 10.0 ** ((a1 + a2) - (apiv1 + apiv2))

        return y[0] if scalar else y


class tsbpl(Additive):
    """Triple smoothly broken power law (four segments, three smooth breaks)."""

    def __init__(self):
        """Initialise triple sbpl with four indices and three breaks."""

        self.expr = 'tsbpl'
        self.comment = 'triple smoothly broken power-law model'

        self.config = OrderedDict()
        self.config['pivot'] = Cfg(1.0)

        self.params = OrderedDict()
        self.params[r'$\alpha_1$'] = Par(0, unif(-10, 10))
        self.params[r'$\alpha_2$'] = Par(0, unif(-10, 10))
        self.params[r'$\alpha_3$'] = Par(0, unif(-10, 10))
        self.params[r'$\alpha_4$'] = Par(0, unif(-10, 10))
        self.params[r'log$x_{b1}$'] = Par(2, unif(0, 4))
        self.params[r'log$x_{b2}$'] = Par(2, unif(0, 4))
        self.params[r'log$x_{b3}$'] = Par(2, unif(0, 4))
        self.params[r'$\delta_1$'] = Par(0.3, unif(1e-3, 3))
        self.params[r'$\delta_2$'] = Par(0.3, unif(1e-3, 3))
        self.params[r'$\delta_3$'] = Par(0.3, unif(1e-3, 3))
        self.params[r'log$A$'] = Par(0, unif(-10, 10))

    @staticmethod
    def _log_cosh(q):
        r"""Return :math:`\ln\cosh q`, computed stably via :func:`numpy.logaddexp`.

        Equivalent to ``log(cosh(q))`` but avoids overflow for large ``|q|``.

        Args:
            q: Array of dimensionless log-scale distances from a break.

        Returns:
            Array of :math:`\ln\cosh q` with the same shape as ``q``.
        """

        return np.logaddexp(q, -q) - np.log(2.0)

    def func(self, X):
        r"""Return the triple smoothly broken power law on ``X[:, 0]``.

        Four power-law segments joined by three smooth breaks
        (:math:`x_{b1}, x_{b2}, x_{b3}`): the local index runs
        :math:`\alpha_1 \to \alpha_2 \to \alpha_3 \to \alpha_4` with mean
        slope :math:`b = (\alpha_1 + \alpha_4)/2`, each break smoothed by its
        own :math:`\delta_i`. The curve is normalised to the amplitude
        :math:`A` at the ``pivot`` config value.

        Args:
            X: Scalar, 1-D, or 2-D input; the x-grid is taken from column 0
                and must be strictly positive.

        Returns:
            Model values matching the input sampling (scalar in, scalar out).

        Raises:
            ValueError: If any ``x`` is non-positive (the model lives on a
                log scale).
        """

        x, scalar = self.asx(X)

        if np.any(x <= 0):
            raise ValueError('tsbpl requires positive x (evaluated on a log scale)')

        xpiv = self.config['pivot'].value

        alpha1 = self.params[r'$\alpha_1$'].value
        alpha2 = self.params[r'$\alpha_2$'].value
        alpha3 = self.params[r'$\alpha_3$'].value
        alpha4 = self.params[r'$\alpha_4$'].value
        xb1 = 10 ** self.params[r'log$x_{b1}$'].value
        xb2 = 10 ** self.params[r'log$x_{b2}$'].value
        xb3 = 10 ** self.params[r'log$x_{b3}$'].value
        delta1 = self.params[r'$\delta_1$'].value
        delta2 = self.params[r'$\delta_2$'].value
        delta3 = self.params[r'$\delta_3$'].value
        amp = 10 ** self.params[r'log$A$'].value

        b = 0.5 * (alpha1 + alpha4)
        m1 = 0.5 * (alpha2 - alpha1)
        m2 = 0.5 * (alpha3 - alpha2)
        m3 = 0.5 * (alpha4 - alpha3)

        q1 = np.log10(x / xb1) / delta1
        q2 = np.log10(x / xb2) / delta2
        q3 = np.log10(x / xb3) / delta3

        qpiv1 = np.log10(xpiv / xb1) / delta1
        qpiv2 = np.log10(xpiv / xb2) / delta2
        qpiv3 = np.log10(xpiv / xb3) / delta3

        a1 = m1 * delta1 * self._log_cosh(q1)
        a2 = m2 * delta2 * self._log_cosh(q2)
        a3 = m3 * delta3 * self._log_cosh(q3)

        apiv1 = m1 * delta1 * self._log_cosh(qpiv1)
        apiv2 = m2 * delta2 * self._log_cosh(qpiv2)
        apiv3 = m3 * delta3 * self._log_cosh(qpiv3)

        shape_term = (a1 + a2 + a3) - (apiv1 + apiv2 + apiv3)
        y = amp * (x / xpiv) ** b * 10.0**shape_term

        return y[0] if scalar else y


class spindown(Additive):
    """Magnetar spin-down luminosity model solved via an ODE for the spin frequency."""

    # Physical constants (CGS) and ODE rescaling factors -- parameter-independent.
    _R = 1.2e6  # neutron-star radius [cm]
    _C = 2.998e10  # speed of light [cm/s]
    _G = 6.674e-8  # gravitational constant [cgs]
    _I = 3.33e45  # moment of inertia [g cm^2]
    _EPS = 1.0e-4  # ellipticity
    _FB = 0.1  # beaming factor
    _T = 1e2  # time rescaling [s]
    _Y = 1e3  # frequency rescaling [rad/s]

    # Gravitational-wave braking coefficient does not depend on any parameter.
    _B_GW = -32 * _G * _I * _EPS**2 / (5 * _C**5)
    _B_HAT = _B_GW * _T * _Y**4

    def __init__(self):
        """Initialise spin-down model with log-B, log-P0, and log-efficiency parameters."""
        super().__init__()

        self.expr = 'spindown'
        self.comment = 'magnetar spin-down model'

        self.config = OrderedDict()

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
        using the physical magnetar constants stored as class attributes, then
        converts the resulting spin frequency to radiated power.

        Args:
            X: 2-D input array; the time grid (in seconds) is taken from column 0.

        Returns:
            1-D array of luminosity values (in erg/s) with the same length as ``X``.

        Raises:
            ValueError: If any requested time is negative (integration starts
                at ``t = 0``).
            RuntimeError: If the ODE solver fails to converge.
        """

        t, scalar = self.asx(X)

        Bp = 10 ** self.params['log$B_{p,15}$'].value * 1e15
        P0 = 10 ** self.params['log$P_{0,-3}$'].value * 1e-3
        eta = 10 ** self.params['log$\\eta_{-3}$'].value * 1e-3

        a = -(Bp**2) * self._R**6 / (6 * self._C**3 * self._I)
        a_hat = a * self._T * self._Y**2

        y0 = 2.0 * np.pi / P0
        y0_hat = y0 / self._Y

        t_hat = np.asarray(t, dtype=float) / self._T

        if np.any(t_hat < 0):
            raise ValueError('spindown requires non-negative times (integration starts at t=0)')

        # solve_ivp needs t_eval sorted within the span; sort, solve, then restore order.
        order = np.argsort(t_hat, kind='stable')
        t_sorted = t_hat[order]
        tf = t_sorted[-1]

        if tf == 0:
            # Every requested time is t=0, i.e. the initial condition Omega(0) = y0.
            Omega = np.full(t_hat.shape, y0)
        else:
            y_hat = solve_ivp(
                self.ode, (0.0, tf), [y0_hat], args=(a_hat, self._B_HAT), t_eval=t_sorted
            )
            if not y_hat.success:
                raise RuntimeError('ODE solver failed: ' + y_hat.message)

            Omega = np.empty_like(t_hat)
            Omega[order] = y_hat.y[0] * self._Y

        luminosity = eta / self._FB * Bp**2 * self._R**6 * Omega**4 / (6 * self._C**3)

        return luminosity[0] if scalar else luminosity


class psd(Additive):
    """Power spectral density model: white noise plus up to two Lorentzian components."""

    def __init__(self):
        """Initialise the PSD model for the requested variant.

        Args:
            expr: Model variant string; one of ``'white'``, ``'psd1'``, or
                ``'psd2'``. ``'white'`` is flat noise only; ``'psd1'`` adds
                one Lorentzian peak; ``'psd2'`` adds two Lorentzian peaks.

        Raises:
            ValueError: If ``expr`` is not one of the supported variant strings.
        """

        self.expr = 'psd'
        self.comment = 'power spectral distribution model'

        self.config = OrderedDict()
        self.config['psd_num'] = Cfg(1)

    @cached_property(lambda self: self.config['psd_num'].value)
    def params(self):

        params = OrderedDict()

        if self.config['psd_num'].value == 0:
            params[r'$A_w$'] = Par(2, unif(0, 5))

        elif self.config['psd_num'].value == 1:
            params[r'$A_w$'] = Par(2, unif(0, 5))
            params[r'$A_1$'] = Par(10, unif(0, 30))
            params[r'$\nu_1$'] = Par(1000, unif(500, 5000))
            params[r'log$\Delta\nu_1$'] = Par(2, unif(1, 3))

        elif self.config['psd_num'].value == 2:
            params[r'$A_w$'] = Par(2, unif(0, 5))
            params[r'$A_1$'] = Par(10, unif(0, 30))
            params[r'$\nu_1$'] = Par(1000, unif(500, 5000))
            params[r'log$\Delta\nu_1$'] = Par(2, unif(1, 3))
            params[r'$A_2$'] = Par(10, unif(0, 30))
            params[r'$\nu_2$'] = Par(1000, unif(500, 5000))
            params[r'log$\Delta\nu_2$'] = Par(2, unif(1, 3))

        else:
            raise ValueError('Invalid value for psd_num config.')

        return params

    def func(self, X):
        """Return the PSD model value at each frequency in ``X[:, 0]``.

        Args:
            X: 2-D input array; the frequency grid is taken from column 0.

        Returns:
            1-D array of PSD values with the same length as ``X``.

        Raises:
            ValueError: If ``self.expr`` is not a supported variant string.
        """

        nu, scalar = self.asx(X)

        psd_num = self.config['psd_num'].value
        Aw = self.params[r'$A_w$'].value

        if psd_num == 0:
            Pnu = np.ones_like(nu, dtype=float) * Aw

        elif psd_num == 1:
            A1 = self.params[r'$A_1$'].value
            nu1 = self.params[r'$\nu_1$'].value
            dnu1 = 10 ** self.params[r'log$\Delta\nu_1$'].value
            Pnu = Aw + A1 / (1 + (nu - nu1) ** 2 / dnu1**2)

        elif psd_num == 2:
            A1 = self.params[r'$A_1$'].value
            nu1 = self.params[r'$\nu_1$'].value
            dnu1 = 10 ** self.params[r'log$\Delta\nu_1$'].value
            A2 = self.params[r'$A_2$'].value
            nu2 = self.params[r'$\nu_2$'].value
            dnu2 = 10 ** self.params[r'log$\Delta\nu_2$'].value
            Pnu = Aw + A1 / (1 + (nu - nu1) ** 2 / dnu1**2) + A2 / (1 + (nu - nu2) ** 2 / dnu2**2)

        return Pnu[0] if scalar else Pnu
