"""Matplotlib plotting helpers for models, inference results, and posteriors.

``Plot`` provides static methods that produce ``matplotlib.figure.Figure``
objects from curvefit model and inference objects.  All renderers use
matplotlib only -- there is no plotly dependency in this module.
"""

import matplotlib.pyplot as plt
import numpy as np


class Plot:
    """Static factory for figures over curvefit models and inference results.

    Every method accepts a curvefit object and returns a
    ``matplotlib.figure.Figure``.  No backend selection is needed; the
    class always uses matplotlib.
    """

    @staticmethod
    def model(model, x, ax=None):
        """Plot the model curve over a user-supplied x grid.

        Evaluates ``model.func`` at every point in ``x`` and draws the
        resulting curve, labelling the line with ``model.expr``.

        Args:
            model: A curvefit ``Model`` instance whose ``func`` and ``expr``
                attributes are used.
            x: 1-D array-like of x values at which the model is evaluated.
            ax: Optional existing ``matplotlib.axes.Axes`` to draw into.
                A new figure and axes are created when ``None``.

        Returns:
            The ``matplotlib.figure.Figure`` containing the model curve.
        """

        x = np.asarray(x, dtype=float)
        y = model.func(x[:, None])

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure

        ax.plot(x, y, lw=1.5, label=model.expr)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()

        return fig

    @staticmethod
    def infer(post, nsample=200, ngrid=200, seed=42):
        """Plot data with the posterior best-fit curve and 1-sigma band.

        The top panel shows, for each ``Pair`` in the posterior, the observed
        data points with asymmetric error bars and upper-limit markers
        (downward-pointing triangles for points where ``unit.up`` is
        ``True``), the best-fit model curve evaluated at ``par_best_ci``, and
        a shaded 1-sigma posterior band derived from ``nsample`` random
        draws from the posterior sample.  The bottom panel shows per-point
        residuals in units of sigma against a reference line at zero.

        Args:
            post: A curvefit ``Posterior`` instance that exposes
                ``Pair``, ``posterior_sample``, ``par_best_ci``,
                ``free_nparams``, and ``at_par``.
            nsample: Maximum number of posterior samples drawn to build the
                credible band.  Fewer samples are used when the chain is
                shorter than this value.
            ngrid: Number of evenly spaced points on the x grid used to
                evaluate the band curves.
            seed: Integer seed for the random-number generator used to
                select the sample subset.

        Returns:
            The ``matplotlib.figure.Figure`` containing the top data/fit
            panel and the bottom residual panel.
        """

        fig, (ax, axr) = plt.subplots(
            2, 1, sharex=True, figsize=(7, 6), gridspec_kw={'height_ratios': [3, 1]}
        )

        xmins, xmaxs = list(), list()
        for pair in post.Pair:
            for unit in pair.data.data.values():
                xmins.append(np.min(unit.x))
                xmaxs.append(np.max(unit.x))
        grid = np.linspace(min(xmins), max(xmaxs), ngrid)

        post.at_par(post.par_best_ci)
        best = {id(pair): pair.mo_func(grid, pair.pvalues) for pair in post.Pair}

        rng = np.random.default_rng(seed)
        samples = post.posterior_sample
        size = min(nsample, samples.shape[0])
        idx = rng.choice(samples.shape[0], size=size, replace=False)

        bands = dict()
        for pair in post.Pair:
            curves = list()
            for k in idx:
                post.at_par(samples[k, : post.free_nparams])
                curves.append(pair.mo_func(grid, pair.pvalues))
            bands[id(pair)] = np.quantile(np.array(curves), [0.16, 0.84], axis=0)

        post.at_par(post.par_best_ci)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for ci, pair in enumerate(post.Pair):
            color = colors[ci % len(colors)]

            ax.plot(grid, best[id(pair)], color=color, lw=1.5)
            lo, hi = bands[id(pair)]
            ax.fill_between(grid, lo, hi, color=color, alpha=0.3)

            for unit in pair.data.data.values():
                norm = ~unit.up
                ax.errorbar(
                    unit.x[norm],
                    unit.y[norm],
                    yerr=[unit.yerr[0][norm], unit.yerr[1][norm]],
                    xerr=[unit.xerr[0][norm], unit.xerr[1][norm]],
                    fmt='o',
                    color=color,
                    ms=3,
                    lw=1,
                    capsize=0,
                )

                if np.any(unit.up):
                    ax.errorbar(
                        unit.x[unit.up],
                        unit.y[unit.up],
                        yerr=0.1 * np.abs(unit.y[unit.up]),
                        uplims=True,
                        fmt='v',
                        color=color,
                        ms=6,
                    )

                my = pair.mo_func(unit.x, pair.pvalues)
                sigma = np.where(unit.y < my, unit.yerr[1], unit.yerr[0])
                axr.errorbar(
                    unit.x[norm],
                    ((unit.y - my) / sigma)[norm],
                    yerr=1,
                    fmt='o',
                    color=color,
                    ms=3,
                    lw=1,
                    capsize=0,
                )

        axr.axhline(0, color='gray', lw=1, ls='--')
        ax.set_ylabel('y')
        axr.set_ylabel('residual')
        axr.set_xlabel('x')
        fig.tight_layout()

        return fig

    @staticmethod
    def post_corner(post, **kwargs):
        """Produce a corner plot of the free-parameter posterior samples.

        Draws a triangle (corner) plot of the joint and marginal posterior
        distributions for all free parameters using the ``corner`` library.
        Extra keyword arguments are forwarded directly to
        ``corner.corner``.

        Args:
            post: A curvefit ``Posterior`` instance that exposes
                ``posterior_sample`` and ``free_plabels``.
            **kwargs: Additional keyword arguments passed through to
                ``corner.corner`` (e.g. ``labels``, ``truths``,
                ``quantiles``).

        Returns:
            The ``matplotlib.figure.Figure`` produced by ``corner.corner``.
        """

        import corner

        samples = post.posterior_sample[:, : post.free_nparams]
        labels = post.free_plabels

        return corner.corner(samples, labels=labels, **kwargs)
