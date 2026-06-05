import numpy as np

from curvefit.data.data import Data, DataUnit
from curvefit.infer.analyzer import Bootstrap, Posterior, SampleAnalyzer
from curvefit.infer.infer import BayesInfer, MaxLikeFit
from curvefit.model.local import line


def _infer(cls):
    x = np.linspace(0, 10, 20)
    y = 2.0 * x + 1.0
    unit = DataUnit(x, y, yerr=np.full(x.size, 0.3), stat='chi2')
    data = Data([('d', unit)])
    model = line()
    model.params['logv'].frozen = True
    return cls([(data, model)])


def test_bootstrap_from_sample_matrix():
    infer = _infer(MaxLikeFit)
    infer._you_free()
    nfree = infer.free_nparams
    # fabricate a bootstrap sample: first row is the best fit
    rng = np.random.default_rng(0)
    best = np.array([2.0, 1.0])
    draws = best + rng.normal(0, 0.05, size=(50, nfree))
    draws[0] = best
    loglike = np.array([infer.calc_loglike(t) for t in draws])
    infer.bootstrap_sample = np.hstack((draws, loglike[:, None]))

    boot = Bootstrap(infer)
    assert isinstance(boot, SampleAnalyzer)
    # best is the first row
    assert np.allclose(boot.par_best, [2.0, 1.0])
    assert len(boot.par_median) == nfree
    ci = boot.par_interval(0.6827)
    assert len(ci) == nfree


def test_posterior_still_works():
    infer = _infer(BayesInfer)
    infer._you_free()
    nfree = infer.free_nparams
    rng = np.random.default_rng(1)
    draws = np.array([2.0, 1.0]) + rng.normal(0, 0.05, size=(200, nfree))
    loglike = np.array([infer.calc_loglike(t) for t in draws])
    infer.posterior_sample = np.hstack((draws, loglike[:, None]))

    post = Posterior(infer)
    assert len(post.par_best_ci) == nfree
    assert np.all(np.isfinite(post.par_mean))
