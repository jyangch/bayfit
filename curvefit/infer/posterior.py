"""Posterior and bootstrap sample analysers over a fitted :class:`Infer`.

``SampleAnalyzer`` absorbs an ``Infer``, loads its ``(nsample, nfree+1)``
sample matrix (parameters plus a trailing log-probability column), attaches
a :class:`~curvefit.util.post.Post` to each free parameter, and exposes
point estimates, credible intervals, and information criteria. ``Posterior``
reads ``posterior_sample`` (nested sampling / MCMC); ``Bootstrap`` reads
``bootstrap_sample`` (maximum-likelihood resampling).
"""

from collections import OrderedDict

import numpy as np

from ..util.info import Info
from ..util.post import Post
from .infer import Infer


class SampleAnalyzer(Infer):
    """Shared analysis over a parameter sample absorbed from an ``Infer``.

    Subclasses set :attr:`sample_attribute` to the source ``Infer``
    attribute holding the sample matrix and :attr:`analyzer_type` to the
    display label.

    Attributes:
        sample_attribute: Name of the source attribute (set by subclass).
        analyzer_type: Human-readable label shown in ``__str__``.
    """

    sample_attribute = None
    analyzer_type = 'Sample Analysis Results'

    def __init__(self, infer):
        """Absorb ``infer`` and attach per-parameter posteriors.

        Args:
            infer: A fitted ``Infer`` carrying the sample matrix named by
                :attr:`sample_attribute`.
        """

        self.infer = infer

    @property
    def infer(self):

        return self._infer

    @infer.setter
    def infer(self, new_infer):
        """Validate, copy the infer state, load the sample, and analyse.

        Raises:
            TypeError: If ``new_infer`` is not an ``Infer``.
        """

        if not isinstance(new_infer, Infer):
            raise TypeError('expected an instance of Infer')

        self._infer = new_infer
        self.__dict__.update(new_infer.__dict__)

        self._load_sample()
        self._post()

    def _load_sample(self):
        """Load and validate the sample matrix from :attr:`sample_attribute`.

        Raises:
            AttributeError: If the attribute is undefined or absent.
            ValueError: If the matrix is not 2D with ``nfree + 1`` columns.
        """

        if self.sample_attribute is None:
            raise AttributeError('sample_attribute is not defined')

        self.sample = getattr(self, self.sample_attribute, None)
        if self.sample is None:
            raise AttributeError(f'{self.sample_attribute} is not available')

        self.sample = np.asarray(self.sample, dtype=float)
        if self.sample.ndim != 2:
            raise ValueError(f'{self.sample_attribute} is expected to be a 2D array')
        if self.sample.shape[1] != self.free_nparams + 1:
            raise ValueError(
                f'{self.sample_attribute} is expected to have {self.free_nparams + 1} columns'
            )

    def _post(self):
        """Attach a :class:`Post` to each free parameter and set the best fit."""

        for i in range(self.free_nparams):
            sample = self.sample[:, i].copy()
            logprob = self.sample[:, -1].copy()

            self.free_par[i + 1].post = Post(sample, logprob)

        self._set_best()

    def _set_best(self):
        """Set each free parameter's ``post.best_ci``; overridden per subclass."""

        raise NotImplementedError

    @property
    def par_mean(self):
        """Per-free-parameter posterior summaries.

        The ``par_mean``/``par_median``/``par_best``/``par_best_ci`` family
        each returns one value per free parameter, read from its ``Post``.
        """

        return [par.post.mean for par in self.free_par.values()]

    @property
    def par_median(self):

        return [par.post.median for par in self.free_par.values()]

    @property
    def par_best(self):

        return [par.post.best for par in self.free_par.values()]

    @property
    def par_best_ci(self):

        return [par.post.best_ci for par in self.free_par.values()]

    def par_quantile(self, q):
        """Per-parameter quantile at probability ``q``."""

        return [par.post.quantile(q) for par in self.free_par.values()]

    def par_interval(self, q):
        """Per-parameter central credible interval at probability ``q``."""

        return [par.post.interval(q) for par in self.free_par.values()]

    @property
    def par_Isigma(self):
        """Per-parameter 1/2/3-sigma central intervals."""

        return [par.post.Isigma for par in self.free_par.values()]

    @property
    def par_IIsigma(self):

        return [par.post.IIsigma for par in self.free_par.values()]

    @property
    def par_IIIsigma(self):

        return [par.post.IIIsigma for par in self.free_par.values()]

    def par_error(self, par, q=0.6827):
        """Return ``[low_gap, high_gap]`` per parameter around point estimate ``par``."""

        ci = self.par_interval(q)

        return [np.diff([c[0], p, c[1]]).tolist() for p, c in zip(par, ci, strict=False)]

    @property
    def max_loglike(self):
        """Log-likelihood at the best-fit parameter vector."""

        self.at_par(self.par_best)

        return self.loglike

    @property
    def aic(self):
        """Akaike information criterion at the best fit."""

        return -2 * self.max_loglike + 2 * self.free_nparams

    @property
    def aicc(self):
        """Small-sample corrected AIC."""

        return self.aic + 2 * self.free_nparams * (self.free_nparams + 1) / (
            self.npoint - self.free_nparams - 1
        )

    @property
    def bic(self):
        """Bayesian information criterion at the best fit."""

        return -2 * self.max_loglike + self.free_nparams * np.log(self.npoint)

    @property
    def lnZ(self):
        """Log-evidence if a sampler stored one, else ``None``."""

        try:
            return self.logevidence
        except AttributeError:
            return None

    @property
    def free_par_info(self):
        """Free-parameter summary table (mean/median/best/1-sigma CI)."""

        self._you_free()

        free_par_info = Info.list_dict_to_dict(self.free_params)

        del free_par_info['Posterior']
        del free_par_info['Mates']
        del free_par_info['Frozen']
        del free_par_info['Prior']
        del free_par_info['Value']

        free_par_info['Mean'] = [f'{par:.3f}' for par in self.par_mean]
        free_par_info['Median'] = [f'{par:.3f}' for par in self.par_median]
        free_par_info['Best'] = [f'{par:.3f}' for par in self.par_best_ci]
        free_par_info['1sigma CI'] = [
            '[{:.3f}, {:.3f}]'.format(*tuple(ci)) for ci in self.par_Isigma
        ]

        return Info.from_dict(free_par_info)

    @property
    def stat_info(self):
        """Goodness-of-fit table evaluated at the best fit."""

        self.at_par(self.par_best_ci)

        return Info.from_dict(self.all_stat)

    @property
    def IC_info(self):
        """Information-criteria table (AIC/AICc/BIC/lnZ)."""

        IC_info = OrderedDict()
        IC_info['AIC'] = [f'{self.aic:.2f}']
        IC_info['AICc'] = [f'{self.aicc:.2f}']
        IC_info['BIC'] = [f'{self.bic:.2f}']
        IC_info['lnZ'] = [f'{self.lnZ}' if self.lnZ is None else f'{self.lnZ:.2f}']

        return Info.from_dict(IC_info)

    def __str__(self):

        print(self.free_par_info.table)
        print(self.stat_info.table)
        print(self.IC_info.table)

        return ''


class Posterior(SampleAnalyzer):
    """Posterior analyser for nested-sampling / MCMC ``posterior_sample``."""

    sample_attribute = 'posterior_sample'
    analyzer_type = 'Posterior Results'

    def _set_best(self):
        """Pick the highest-logprob draw lying inside every 1-sigma interval."""

        argsort = np.argsort(self.sample[:, -1])[::-1]
        sort_sample = self.sample[:, 0:-1].copy()[argsort]

        for sample in sort_sample:
            if np.array(
                [(ci[0] <= sample[i] <= ci[1]) for i, ci in enumerate(self.par_interval(0.6827))]
            ).all():
                for par, value in zip(self.free_par.values(), sample, strict=False):
                    par.post.best_ci = value

                break

    @property
    def posterior_statistic(self):
        """Mean/median and 1/2/3-sigma quantiles of the full posterior matrix."""

        mean = np.mean(self.sample, axis=0)
        median = np.median(self.sample, axis=0)

        q = 68.27 / 100
        Isigma = np.quantile(self.sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        q = 95.45 / 100
        IIsigma = np.quantile(self.sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        q = 99.73 / 100
        IIIsigma = np.quantile(self.sample, [0.5 - q / 2, 0.5 + q / 2], axis=0)

        return dict(
            [
                ('mean', mean),
                ('median', median),
                ('Isigma', Isigma),
                ('IIsigma', IIsigma),
                ('IIIsigma', IIIsigma),
            ]
        )


class Bootstrap(SampleAnalyzer):
    """Bootstrap analyser for maximum-likelihood ``bootstrap_sample``.

    The first row of the sample is the best-fit vector produced by
    :class:`~curvefit.infer.infer.MaxLikeFit`.
    """

    sample_attribute = 'bootstrap_sample'
    analyzer_type = 'Bootstrap Results'

    def _set_best(self):
        """Use the first sample row (the stored best fit) as ``best_ci``.

        Note:
            Deliberate departure from bayspec, which records the best fit in
            ``post.truth`` and derives ``best_ci`` by interval containment.
            Here the maximum-likelihood point is the best estimate, so it is
            stored directly as ``best_ci``.
        """

        best = self.sample[0, 0:-1]

        for par, value in zip(self.free_par.values(), best, strict=False):
            par.post.best_ci = value
