import matplotlib.pyplot as plt
import numpy as np


class Plot:
    @staticmethod
    def model(model, x, ax=None):

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

        import corner

        samples = post.posterior_sample[:, : post.free_nparams]
        labels = post.free_plabels

        return corner.corner(samples, labels=labels, **kwargs)
