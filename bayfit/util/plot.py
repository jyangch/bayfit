"""Matplotlib plotting helpers for models, inference results, and posteriors.

``Plot`` provides static methods that produce ``matplotlib.figure.Figure``
objects from bayfit model and inference objects.  All renderers use
matplotlib only -- there is no plotly dependency in this module.
"""

import sys
import warnings

import corner
from getdist import MCSamples, plots
import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from ..data.data import Data, DataUnit
from ..infer.analyzer import Bootstrap, Posterior
from ..infer.infer import BayesInfer, Infer
from ..infer.pair import Pair
from ..model.model import Model
from .corner import corner_plotly
from .tools import json_dump


class Plot:
    """Static factory for figures over bayfit models and inference results.

    Every method accepts a bayfit object and returns a
    ``matplotlib.figure.Figure``.  No backend selection is needed; the
    class always uses matplotlib.
    """

    colors = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.G10
        + px.colors.qualitative.T10
        + px.colors.qualitative.Alphabet
    )

    @staticmethod
    def get_rgb(color, opacity=1.0):
        """Convert a matplotlib color plus opacity into a Plotly ``rgba`` string."""

        rgba = mpl.colors.to_rgba(color)
        r, g, b = (int(x * 255) for x in rgba[:3])

        return f'rgba({r}, {g}, {b}, {opacity:f})'

    @staticmethod
    def dataunit(cls, ploter='plotly', xlog=False, ylog=False):
        """Plot a single ``DataUnit``'s observed data.

        Args:
            cls: ``DataUnit`` to plot; must be complete.
            ploter: Backend -- ``'plotly'`` or ``'matplotlib'``.
            xlog: If ``True``, use a logarithmic scale for the x-axis.
            ylog: If ``True``, use a logarithmic scale for the y-axis.

        Returns:
            A :class:`Figure` wrapping the plot.

        Raises:
            TypeError: If ``cls`` is not a ``DataUnit``.
            AttributeError: If the ``DataUnit`` fails its completeness check.
        """

        if not isinstance(cls, DataUnit):
            raise TypeError('cls is not DataUnit type, cannot call dataunit method')

        xs = cls.xs[:, 0].astype(float)
        xerr = cls.xerr[0].astype(float)

        ys = cls.ys.astype(float)
        yerr = cls.yerr.astype(float)

        if ploter == 'plotly':
            fig = go.Figure()
            unit = go.Scatter(
                x=xs,
                y=ys,
                mode='markers',
                showlegend=True,
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=xerr[1],
                    arrayminus=xerr[0],
                    thickness=1.5,
                    width=0,
                ),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=yerr[1],
                    arrayminus=yerr[0],
                    thickness=1.5,
                    width=0,
                ),
                marker=dict(symbol='circle', size=3),
            )

            fig.add_trace(unit)

            fig.update_xaxes(title_text='X', type='log' if xlog else 'linear')
            fig.update_yaxes(title_text='Y', type='log' if ylog else 'linear')
            fig.update_layout(template='plotly_white', height=600, width=600)
            fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            fig = plt.figure(figsize=(7, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])

            ax.errorbar(
                xs,
                ys,
                xerr=xerr,
                yerr=yerr,
                fmt='none',
                ecolor=Plot.colors[0],
                elinewidth=1.0,
                capsize=0,
                label='Source',
            )
            ax.set_xscale('log' if xlog else 'linear')
            ax.set_yscale('log' if ylog else 'linear')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.minorticks_on()
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='x', which='both', labeltop=False, labelbottom=True)
            ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
            ax.legend()

        fig_data = {'xs': xs, 'ys': ys, 'xerr': xerr, 'yerr': yerr}

        return Figure(fig, fig_data, ploter)

    @staticmethod
    def data(cls, ploter='plotly', logx=False, logy=False):
        """Plot every ``DataUnit`` in a ``Data`` container on one figure.

        Args:
            cls: ``Data`` whose units are drawn together.
            ploter: Backend -- ``'plotly'`` or ``'matplotlib'``.
            logx: If ``True``, use a logarithmic scale for the x-axis.
            logy: If ``True``, use a logarithmic scale for the y-axis.

        Returns:
            A :class:`Figure` wrapping the plot.

        Raises:
            TypeError: If ``cls`` is not a ``Data``.
        """

        if not isinstance(cls, Data):
            raise TypeError('cls is not Data type, cannot call data method')

        if ploter == 'plotly':
            fig = go.Figure()

        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42
            fig = plt.figure(figsize=(7, 6))
            gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(gs[0, 0])

        xs = [x[:, 0] for x in cls.xs]
        xerr = [xe[0] for xe in cls.xerr]
        ys = cls.ys
        yerr = cls.yerr

        fig_data = {}

        for i, name in enumerate(cls.names):
            if ploter == 'plotly':
                unit = go.Scatter(
                    x=xs[i].astype(float),
                    y=ys[i].astype(float),
                    mode='markers',
                    name=f'{name}',
                    showlegend=True,
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=xerr[i][1].astype(float),
                        arrayminus=xerr[i][0].astype(float),
                        color=Plot.colors[i],
                        thickness=1.5,
                        width=0,
                    ),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=yerr[i][1].astype(float),
                        arrayminus=yerr[i][0].astype(float),
                        color=Plot.colors[i],
                        thickness=1.5,
                        width=0,
                    ),
                    marker=dict(symbol='circle', size=3, color=Plot.colors[i]),
                )
                fig.add_trace(unit)

            elif ploter == 'matplotlib':
                ax.errorbar(
                    xs[i],
                    ys[i],
                    xerr=xerr[i],
                    yerr=yerr[i],
                    fmt='none',
                    ecolor=Plot.colors[i],
                    elinewidth=0.8,
                    capsize=0,
                    capthick=0,
                    label=name,
                )

            fig_data[name] = {
                'xs': xs[i],
                'ys': ys[i],
                'xerr': xerr[i],
                'yerr': yerr[i],
            }

        if ploter == 'plotly':
            fig.update_xaxes(title_text='X', type='log' if logx else 'linear')
            fig.update_yaxes(title_text='Y', type='log' if logy else 'linear')
            fig.update_layout(template='plotly_white', height=600, width=600)
            fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        elif ploter == 'matplotlib':
            ax.set_xscale('log' if logx else 'linear')
            ax.set_yscale('log' if logy else 'linear')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.minorticks_on()
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='x', which='both', labeltop=False, labelbottom=True)
            ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
            ax.legend()

        return Figure(fig, fig_data, ploter)

    @staticmethod
    def model(ploter='plotly', xlog=False, ylog=False, post=False):
        """Create an empty :class:`ModelPlot` for accumulating model traces.

        Args:
            ploter: Backend -- ``'plotly'`` or ``'matplotlib'``.
            xlog: If ``True``, use a logarithmic scale for the x-axis.
            ylog: If ``True``, use a logarithmic scale for the y-axis.
            post: If ``True``, also draw the posterior credible band.

        Returns:
            A fresh :class:`ModelPlot` ready for ``add_model`` calls.
        """

        modelplot = ModelPlot(ploter=ploter, xlog=xlog, ylog=ylog, post=post)

        return modelplot

    @staticmethod
    def pair(cls, ploter='plotly', xlog=False, ylog=False):
        """Plot data and model together for a ``Pair``, with residual panel.

        The top panel shows observed and model curves for every data unit;
        the bottom panel shows residuals in units of sigma.

        Args:
            cls: ``Pair`` whose data and model are drawn.
            ploter: Backend -- ``'plotly'`` or ``'matplotlib'``.
            xlog: If ``True``, use a logarithmic scale for the x-axis.
            ylog: If ``True``, use a logarithmic scale for the y-axis.

        Returns:
            A :class:`Figure` wrapping the plot.

        Raises:
            TypeError: If ``cls`` is not a ``Pair``.
        """

        if not isinstance(cls, Pair):
            raise TypeError('cls is not Pair type, cannot call pair method')

        if ploter == 'plotly':
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.75, 0.25],
                shared_xaxes=True,
                horizontal_spacing=0,
                vertical_spacing=0.02,
            )

        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42
            fig = plt.figure(figsize=(6, 8))
            gs = fig.add_gridspec(4, 1, wspace=0, hspace=0)
            ax1 = fig.add_subplot(gs[0:3, 0])
            ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)

        obs_xs = [x[:, 0] for x in cls.data.xs]
        obs_xerr = [xe[0] for xe in cls.data.xerr]
        obs_ys = cls.data.ys
        obs_yerr = cls.data.yerr
        mo_ys = cls.model.ys
        res_ys = cls.residuals

        fig_data = {}

        for i, name in enumerate(cls.data.names):
            if ploter == 'plotly':
                obs = go.Scatter(
                    x=obs_xs[i].astype(float),
                    y=obs_ys[i].astype(float),
                    mode='markers',
                    name=name,
                    showlegend=False,
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=obs_xerr[i][1].astype(float),
                        arrayminus=obs_xerr[i][0].astype(float),
                        color=Plot.colors[i],
                        thickness=1.5,
                        width=0,
                    ),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=obs_yerr[i][1].astype(float),
                        arrayminus=obs_yerr[i][0].astype(float),
                        color=Plot.colors[i],
                        thickness=1.5,
                        width=0,
                    ),
                    marker=dict(symbol='cross-thin', size=0, color=Plot.colors[i]),
                )
                mo = go.Scatter(
                    x=obs_xs[i].astype(float),
                    y=mo_ys[i].astype(float),
                    name=name,
                    showlegend=True,
                    mode='lines',
                    line=dict(width=2, color=Plot.colors[i]),
                )
                res = go.Scatter(
                    x=obs_xs[i].astype(float),
                    y=res_ys[i].astype(float),
                    name=name,
                    showlegend=False,
                    mode='markers',
                    marker=dict(
                        symbol='cross-thin',
                        size=10,
                        color=Plot.colors[i],
                        line=dict(width=1.5, color=Plot.colors[i]),
                    ),
                )

                fig.add_trace(obs, row=1, col=1)
                fig.add_trace(mo, row=1, col=1)
                fig.add_trace(res, row=2, col=1)

            elif ploter == 'matplotlib':
                ax1.errorbar(
                    obs_xs[i],
                    obs_ys[i],
                    xerr=obs_xerr[i],
                    yerr=obs_yerr[i],
                    fmt='none',
                    ecolor=Plot.colors[i],
                    elinewidth=0.8,
                    capsize=0,
                    capthick=0,
                    label=name,
                )
                ax1.plot(obs_xs[i], mo_ys[i], color=Plot.colors[i], lw=1.0)
                ax2.scatter(
                    obs_xs[i], res_ys[i], marker='+', color=Plot.colors[i], s=40, linewidths=0.8
                )

            fig_data[name] = {
                'obs': {
                    'xs': obs_xs[i],
                    'ys': obs_ys[i],
                    'xerr': obs_xerr[i],
                    'yerr': obs_yerr[i],
                },
                'mo': {'xs': obs_xs[i], 'ys': mo_ys[i]},
                'res': {'xs': obs_xs[i], 'ys': res_ys[i]},
            }

        if ploter == 'plotly':
            fig.update_xaxes(title_text='', row=1, col=1, type='log' if xlog else 'linear')
            fig.update_xaxes(title_text='X', row=2, col=1, type='log' if xlog else 'linear')
            fig.update_yaxes(title_text='Y', row=1, col=1, type='log' if ylog else 'linear')
            fig.update_yaxes(title_text='Sigma', showgrid=False, range=[-3.5, 3.5], row=2, col=1)
            fig.update_layout(template='plotly_white', height=700, width=600)
            fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        elif ploter == 'matplotlib':
            ax1.set_xscale('log' if xlog else 'linear')
            ax1.set_yscale('log' if ylog else 'linear')
            ax1.set_ylabel('Y')
            ax1.minorticks_on()
            ax1.xaxis.set_ticks_position('both')
            ax1.yaxis.set_ticks_position('both')
            ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax1.tick_params(axis='x', which='both', labeltop=False, labelbottom=False)
            ax1.tick_params(axis='y', which='both', labelleft=True, labelright=False)
            ax1.legend()
            ax2.axhline(0, c='grey', lw=1, ls='--')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Sigma')
            ax2.set_ylim([-3.49, 3.49])
            ax2.minorticks_on()
            ax2.xaxis.set_ticks_position('both')
            ax2.yaxis.set_ticks_position('both')
            ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax2.tick_params(axis='x', which='both', labeltop=False, labelbottom=True)
            ax2.tick_params(axis='y', which='both', labelleft=True, labelright=False)

        return Figure(fig, fig_data, ploter)

    @staticmethod
    def emcee_walker(cls):
        """Plot per-parameter emcee walker trajectories.

        Args:
            cls: ``BayesInfer`` or ``Posterior`` exposing
                ``posterior_sample`` and ``free_nparams``.

        Returns:
            A :class:`Figure` wrapping the matplotlib walker plot.

        Raises:
            TypeError: If ``cls`` is not a ``BayesInfer`` or ``Posterior``.
        """

        if not isinstance(cls, (BayesInfer, Posterior)):
            raise TypeError('cls is not BayesInfer or Posterior type, cannot call walker method')

        params_sample = cls.posterior_sample[:, : cls.free_nparams].copy()

        fig, axes = plt.subplots(cls.free_nparams, figsize=(10, 2 * cls.free_nparams), sharex='all')
        for i in range(cls.free_nparams):
            ax = axes[i]
            ax.plot(params_sample[:, :, i], 'k', alpha=0.3)
            ax.set_xlim(0, len(params_sample))
            ax.set_ylabel(cls.free_plabels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ax.minorticks_on()
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='x', which='both', labeltop=False, labelbottom=True)
            ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
        axes[-1].set_xlabel('step number')

        fig_data = None

        return Figure(fig, fig_data, 'matplotlib')

    @staticmethod
    def infer(cls, ploter='plotly', xlog=False, ylog=False, at_par=None):
        """Plot data vs. inferred model (with residuals) from an ``Infer``.

        Args:
            cls: ``Infer`` or one of its subclasses (``Posterior``,
                ``Bootstrap``) to visualize.
            ploter: Backend -- ``'plotly'`` or ``'matplotlib'``.
            xlog: If ``True``, use a logarithmic scale for the x-axis.
            ylog: If ``True``, use a logarithmic scale for the y-axis.
            at_par: Which parameter point to evaluate the model at --
                ``'best'``, ``'best-ci'``, ``'median'``, ``'mean'``, or
                ``'truth'``. Defaults to ``'best'`` for ``Posterior`` and
                ``'truth'`` for ``Bootstrap``.

        Returns:
            A :class:`Figure` wrapping the plot.

        Raises:
            TypeError: If ``cls`` is not an ``Infer``.
            ValueError: If ``at_par`` is not recognized, or if
                ``at_par='truth'`` but some parameters lack a truth value.
        """

        if not isinstance(cls, Infer):
            raise TypeError('cls is not Infer type, cannot call infer method')

        if at_par is None:
            if isinstance(cls, Posterior):
                at_par = 'best'
            if isinstance(cls, Bootstrap):
                at_par = 'truth'

        if isinstance(cls, (Posterior, Bootstrap)):
            if at_par == 'best':
                cls.at_par(cls.par_best)
            elif at_par == 'best-ci':
                cls.at_par(cls.par_best_ci)
            elif at_par == 'median':
                cls.at_par(cls.par_median)
            elif at_par == 'mean':
                cls.at_par(cls.par_mean)
            elif at_par == 'truth':
                if None in cls.par_truth:
                    raise ValueError('no truth value for some parameters')
                else:
                    cls.at_par(cls.par_truth)
            else:
                raise ValueError(f'unsupported at_par argument: {at_par}')

        if isinstance(cls, Bootstrap):
            cls.at_par(cls.par_truth)

        if ploter == 'plotly':
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.75, 0.25],
                shared_xaxes=True,
                horizontal_spacing=0,
                vertical_spacing=0.02,
            )

        elif ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42
            fig = plt.figure(figsize=(6, 8))
            gs = fig.add_gridspec(4, 1, wspace=0, hspace=0)
            ax1 = fig.add_subplot(gs[0:3, 0])
            ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)

        obs_xs = [x[:, 0] for x in cls.data_xs]
        obs_xerr = [xe[0] for xe in cls.data_xerr]
        obs_ys = cls.data_ys
        obs_yerr = cls.data_yerr
        mo_ys = cls.model_ys
        res_ys = cls.residuals

        fig_data = {}

        for i, name in enumerate(cls.data_names):
            if ploter == 'plotly':
                obs = go.Scatter(
                    x=obs_xs[i].astype(float),
                    y=obs_ys[i].astype(float),
                    mode='markers',
                    name=name,
                    showlegend=False,
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=obs_xerr[i][1].astype(float),
                        arrayminus=obs_xerr[i][0].astype(float),
                        color=Plot.colors[i],
                        thickness=1.5,
                        width=0,
                    ),
                    error_y=dict(
                        type='data',
                        array=obs_yerr[i][1].astype(float),
                        arrayminus=obs_yerr[i][0].astype(float),
                        color=Plot.colors[i],
                        thickness=1.5,
                        width=0,
                    ),
                    marker=dict(symbol='cross-thin', size=0, color=Plot.colors[i]),
                )
                mo = go.Scatter(
                    x=obs_xs[i].astype(float),
                    y=mo_ys[i].astype(float),
                    name=name,
                    showlegend=True,
                    mode='lines',
                    line=dict(width=2, color=Plot.colors[i]),
                )
                res = go.Scatter(
                    x=obs_xs[i].astype(float),
                    y=res_ys[i].astype(float),
                    name=name,
                    showlegend=False,
                    mode='markers',
                    marker=dict(
                        symbol='cross-thin',
                        size=10,
                        color=Plot.colors[i],
                        line=dict(width=1.5, color=Plot.colors[i]),
                    ),
                )

                fig.add_trace(obs, row=1, col=1)
                fig.add_trace(mo, row=1, col=1)
                fig.add_trace(res, row=2, col=1)

            elif ploter == 'matplotlib':
                ax1.errorbar(
                    obs_xs[i],
                    obs_ys[i],
                    xerr=[obs_xerr[i][0], obs_xerr[i][1]],
                    yerr=[obs_yerr[i][0], obs_yerr[i][1]],
                    fmt='none',
                    ecolor=Plot.colors[i],
                    elinewidth=0.8,
                    capsize=0,
                    capthick=0,
                    label=name,
                )
                ax1.plot(obs_xs[i], mo_ys[i], color=Plot.colors[i], lw=1.0)
                ax2.scatter(
                    obs_xs[i], res_ys[i], marker='+', color=Plot.colors[i], s=40, linewidths=0.8
                )

            fig_data[name] = {
                'obs': {
                    'xs': obs_xs[i],
                    'ys': obs_ys[i],
                    'xerr': obs_xerr[i],
                    'yerr': obs_yerr[i],
                },
                'mo': {'xs': obs_xs[i], 'ys': mo_ys[i]},
                'res': {'xs': obs_xs[i], 'ys': res_ys[i]},
            }

        if ploter == 'plotly':
            fig.update_xaxes(title_text='', row=1, col=1, type='log' if xlog else 'linear')
            fig.update_xaxes(title_text='X', row=2, col=1, type='log' if xlog else 'linear')
            fig.update_yaxes(title_text='Y', row=1, col=1, type='log' if ylog else 'linear')
            fig.update_yaxes(title_text='Sigma', showgrid=False, range=[-3.5, 3.5], row=2, col=1)
            fig.update_layout(template='plotly_white', height=700, width=600)
            fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        elif ploter == 'matplotlib':
            ax1.set_xscale('log' if xlog else 'linear')
            ax1.set_yscale('log' if ylog else 'linear')
            ax1.set_ylabel('Y')
            ax1.minorticks_on()
            ax1.minorticks_on()
            ax1.xaxis.set_ticks_position('both')
            ax1.yaxis.set_ticks_position('both')
            ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax1.tick_params(axis='x', which='both', labeltop=False, labelbottom=False)
            ax1.tick_params(axis='y', which='both', labelleft=True, labelright=False)
            ax1.legend()
            ax2.axhline(0, c='grey', lw=1, ls='--')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Sigma')
            ax2.set_ylim([-3.49, 3.49])
            ax2.minorticks_on()
            ax2.minorticks_on()
            ax2.xaxis.set_ticks_position('both')
            ax2.yaxis.set_ticks_position('both')
            ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax2.tick_params(axis='x', which='both', labeltop=False, labelbottom=True)
            ax2.tick_params(axis='y', which='both', labelleft=True, labelright=False)

        return Figure(fig, fig_data, ploter)

    @staticmethod
    def post_corner(cls, ploter='plotly', at_par=None):
        """Corner plot of a ``Posterior`` or ``Bootstrap`` sample.

        Args:
            cls: ``Posterior`` or ``Bootstrap`` whose parameter samples are
                visualized.
            ploter: Backend -- ``'plotly'``, ``'getdist'``, or ``'cornerpy'``.
            at_par: Reference point overlaid on the plot -- ``'best'``,
                ``'best-ci'``, ``'median'``, ``'mean'``, or ``'truth'``.
                Defaults to ``'best'`` for ``Posterior`` and ``'truth'``
                for ``Bootstrap``.

        Returns:
            A :class:`Figure` wrapping the plot.

        Raises:
            TypeError: If ``cls`` is not a ``Posterior`` or ``Bootstrap``.
            ValueError: If ``at_par`` is not recognized, if
                ``at_par='truth'`` but some parameters lack a truth value,
                or if ``ploter`` is not one of the supported backends.
        """

        if not isinstance(cls, (Posterior, Bootstrap)):
            raise TypeError('cls is not Posterior or Bootstrap type, cannot call corner method')

        data = cls.param_sample
        weights = np.ones(data.shape[0], dtype=float) / data.shape[0]

        # A non-converged, boundary-pinned posterior can collapse to a near-delta
        # cloud with fewer (distinct) samples than parameters, which corner/getdist
        # cannot plot (they assert n_samples >= n_dims). Skip with a placeholder so
        # a batch loop survives the bad fit instead of crashing on the plot.
        nsample, ndim = data.shape
        if nsample <= ndim or np.ptp(data, axis=0).max() == 0:
            warnings.warn(
                f'Posterior too degenerate to corner-plot ({nsample} samples for '
                f'{ndim} parameters); the run likely did not converge. Returning a '
                f'placeholder figure.',
                stacklevel=2,
            )
            fig = plt.figure(figsize=(4, 4))
            fig.text(0.5, 0.5, 'degenerate posterior\nno corner plot', ha='center', va='center')
            # Tag as matplotlib (not the requested backend) so Figure.save uses
            # fig.savefig: the placeholder is a plain matplotlib figure and has no
            # plotly/getdist export method.
            return Figure(fig, None, 'matplotlib')

        title_fmt = '$%.2f_{-%.2f}^{+%.2f}~(%.2f)$'
        plabels = cls.free_indexed_plabels

        if at_par is None:
            if isinstance(cls, Posterior):
                at_par = 'best'
            if isinstance(cls, Bootstrap):
                at_par = 'truth'

        if at_par == 'best':
            truth = cls.par_best
        elif at_par == 'best-ci':
            truth = cls.par_best_ci
        elif at_par == 'median':
            truth = cls.par_median
        elif at_par == 'mean':
            truth = cls.par_mean
        elif at_par == 'truth':
            if None in cls.par_truth:
                raise ValueError('no truth value for some parameters')
            else:
                truth = cls.par_truth
        else:
            raise ValueError(f'unsupported at_par argument: {at_par}')

        median = cls.par_median
        error = cls.par_error(median)

        if ploter == 'plotly':
            levels = 1.0 - np.exp(-0.5 * np.array([1, 2]) ** 2)

            fig = corner_plotly(
                data, bins=30, weights=weights, smooth1d=2, smooth=2, labels=plabels, levels=levels
            )

            for i in range(cls.free_nparams):
                fig.add_trace(
                    go.Scatter(
                        x=[median[i]],
                        y=[0.01],
                        mode='markers',
                        name=plabels[i],
                        showlegend=False,
                        error_x=dict(
                            type='data',
                            symmetric=False,
                            array=[error[i][1]],
                            arrayminus=[error[i][0]],
                            color='#FF0092',
                            thickness=1,
                            width=0,
                        ),
                        marker=dict(symbol='circle', size=5, color='#FF0092'),
                    ),
                    row=i + 1,
                    col=i + 1,
                )

            for yi in range(cls.free_nparams):
                for xi in range(yi):
                    fig.add_vline(
                        truth[xi], line_width=1, line_color='#FF0092', row=yi + 1, col=xi + 1
                    )
                    fig.add_hline(
                        truth[yi], line_width=1, line_color='#FF0092', row=yi + 1, col=xi + 1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[truth[xi]],
                            y=[truth[yi]],
                            mode='markers',
                            name=f'{plabels[xi]}&{plabels[yi]}',
                            showlegend=False,
                            marker=dict(symbol='square', size=5, color='#FF0092'),
                        ),
                        row=yi + 1,
                        col=xi + 1,
                    )

        elif ploter == 'getdist':
            fig = plots.get_subplot_plotter()
            fig.settings.num_plot_contours = 2
            fig.settings.num_shades = 30
            fig.settings.title_limit_fontsize = 10

            sampler_type = getattr(cls, 'sampler_type', 'mcmc')
            mcsample = MCSamples(samples=data, names=plabels, sampler=sampler_type)
            mcsample.updateSettings({'contours': [0.6827, 0.9545, 0.9973]})

            fig.triangle_plot(mcsample, plabels, shaded=True)

            for i in range(cls.free_nparams):
                ax = fig.subplots[i, i]
                ax.set_title(
                    title_fmt % (median[i], error[i][0], error[i][1], truth[i]),
                    math_fontfamily='stix',
                )
                ax.errorbar(
                    median[i],
                    0.05,
                    xerr=[[error[i][0]], [error[i][1]]],
                    fmt='or',
                    ms=2,
                    ecolor='r',
                    elinewidth=0.7,
                )
                ax.tick_params(axis='both', which='both', zorder=10)

            for yi in range(cls.free_nparams):
                for xi in range(yi):
                    ax = fig.subplots[yi, xi]
                    ax.axvline(truth[xi], color='r', lw=0.7, ls='-')
                    ax.axhline(truth[yi], color='r', lw=0.7, ls='-')
                    ax.scatter(
                        truth[xi], truth[yi], marker='s', color='r', s=10, linewidths=0, zorder=10
                    )
                    ax.tick_params(axis='both', which='both', zorder=10)

        elif ploter == 'cornerpy':
            levels = 1.0 - np.exp(-0.5 * np.array([1, 2]) ** 2)

            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            fig = corner.corner(
                data,
                bins=30,
                color='#08519c',
                weights=weights,
                labels=plabels,
                show_titles=True,
                use_math_text=True,
                smooth1d=2,
                smooth=2,
                levels=levels,
                plot_datapoints=True,
                plot_density=True,
                plot_contours=True,
                fill_contours=False,
                no_fill_contours=False,
            )

            axes = np.array(fig.axes).reshape((cls.free_nparams, cls.free_nparams))

            for i in range(cls.free_nparams):
                ax = axes[i, i]
                ax.set_title(
                    title_fmt % (median[i], error[i][0], error[i][1], truth[i]),
                    math_fontfamily='stix',
                )
                ax.errorbar(
                    median[i],
                    0.005,
                    xerr=[[error[i][0]], [error[i][1]]],
                    fmt='or',
                    ms=2,
                    ecolor='r',
                    elinewidth=1,
                )

            for yi in range(cls.free_nparams):
                for xi in range(yi):
                    ax = axes[yi, xi]
                    ax.axvline(truth[xi], color='r', lw=1, ls='-')
                    ax.axhline(truth[yi], color='r', lw=1, ls='-')
                    ax.scatter(truth[xi], truth[yi], marker='s', color='r', s=20, linewidths=0)

        else:
            raise ValueError(f'unsupported ploter: {ploter}')

        fig_data = None

        return Figure(fig, fig_data, ploter)


class ModelPlot:
    """Accumulating figure that overlays multiple ``Model`` curves.

    Build one via :meth:`Plot.model`, then call :meth:`add_model` for each
    model to compare, and finish with :meth:`get_fig`.

    Attributes:
        ploter: Active backend (``'plotly'`` or ``'matplotlib'``).
        xlog: Whether the x-axis uses a logarithmic scale.
        ylog: Whether the y-axis uses a logarithmic scale.
        post: Whether posterior credible bands are drawn alongside lines.
        fig: Underlying figure object from the chosen backend.
        fig_data: Raw plotted arrays keyed by model expression.
        model_index: Running count of models added; used for color cycling.
    """

    colors = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.G10
        + px.colors.qualitative.T10
        + px.colors.qualitative.Alphabet
    )

    def __init__(self, ploter='plotly', xlog=False, ylog=False, post=False):
        """Initialize the figure for the requested backend.

        Args:
            ploter: ``'plotly'`` or ``'matplotlib'``.
            xlog: If ``True``, use a logarithmic scale for the x-axis.
            ylog: If ``True``, use a logarithmic scale for the y-axis.
            post: If ``True``, draw posterior credible bands.
        """

        self.ploter = ploter
        self.xlog = xlog
        self.ylog = ylog
        self.post = post

        if self.ploter == 'plotly':
            self.fig = go.Figure()

            self.fig.update_xaxes(title_text='X', type='log' if self.xlog else 'linear')
            self.fig.update_yaxes(title_text='Y', type='log' if self.ylog else 'linear')
            self.fig.update_layout(template='plotly_white', height=600, width=600)
            self.fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        elif self.ploter == 'matplotlib':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.size'] = 12
            rcParams['pdf.fonttype'] = 42

            self.fig = plt.figure(figsize=(7, 6))
            gs = self.fig.add_gridspec(1, 1, wspace=0, hspace=0)
            self.ax = self.fig.add_subplot(gs[0, 0])

            self.ax.set_xscale('log' if self.xlog else 'linear')
            self.ax.set_yscale('log' if self.ylog else 'linear')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.minorticks_on()
            self.ax.xaxis.set_ticks_position('both')
            self.ax.yaxis.set_ticks_position('both')
            self.ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            self.ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            self.ax.tick_params(axis='x', which='both', labeltop=False, labelbottom=True)
            self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)

        self.fig_data = {}

        self.model_index = -1

    @staticmethod
    def get_rgb(color, opacity=1.0):
        """Convert a matplotlib color plus opacity into a Plotly ``rgba`` string."""

        rgba = mpl.colors.to_rgba(color)
        r, g, b = (int(x * 255) for x in rgba[:3])

        return f'rgba({r}, {g}, {b}, {opacity:f})'

    def add_model(self, model, X, post=None, at_par=None):
        """Draw ``model`` on the design grid ``X`` onto the accumulating figure.

        When ``post`` is ``True`` the one-sigma credible band is added
        alongside the point estimate.

        Args:
            model: ``Model`` instance to evaluate.
            X: Design grid, shape ``(npoint, 1)``, at which to evaluate the
                model (column ``0`` holds the x values).
            post: Overrides the instance-level ``post`` flag for this call.
            at_par: Which parameter point to evaluate at -- ``'best'``,
                ``'best-ci'``, ``'median'``, ``'mean'``, or ``'truth'``.
                Defaults to ``'best'`` when any truth value is missing,
                otherwise ``'truth'``.

        Raises:
            TypeError: If ``model`` is not a ``Model``.
            ValueError: If ``at_par`` is not recognized, or if
                ``at_par='truth'`` but some parameters lack a truth value.
        """

        if not isinstance(model, Model):
            raise TypeError('model is not Model type, cannot call add_model method')

        if post is None:
            post = self.post

        if post and at_par is None:
            at_par = 'best' if None in model.par_truth else 'truth'

        self.model_index += 1

        xs = np.array(X).astype(float)

        if post:
            if at_par == 'best':
                ys = model.best_func(X).astype(float)
            elif at_par == 'best-ci':
                ys = model.best_ci_func(X).astype(float)
            elif at_par == 'median':
                ys = model.median_func(X).astype(float)
            elif at_par == 'mean':
                ys = model.mean_func(X).astype(float)
            elif at_par == 'truth':
                if None in model.par_truth:
                    raise ValueError('no truth value for some parameters')
                else:
                    ys = model.truth_func(X).astype(float)
            else:
                raise ValueError(f'unsupported at_par argument: {at_par}')
            ys_sample = model.func_sample(X)
            ys_ci = ys_sample['Isigma'].astype(float)
        else:
            ys = model.func(X).astype(float)

        if self.ploter == 'plotly':
            mo = go.Scatter(
                x=xs,
                y=ys,
                mode='lines',
                name=model.expr,
                showlegend=True,
                line=dict(width=2, color=ModelPlot.colors[self.model_index]),
            )
            self.fig.add_trace(mo)

            if post:
                low = go.Scatter(
                    x=xs,
                    y=ys_ci[0],
                    mode='lines',
                    name=f'{model.expr} lower',
                    fill=None,
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                )
                self.fig.add_trace(low)

                upp = go.Scatter(
                    x=xs,
                    y=ys_ci[1],
                    mode='lines',
                    name=f'{model.expr} CI',
                    fill='tonexty',
                    line_color='rgba(0,0,0,0)',
                    fillcolor=ModelPlot.get_rgb(ModelPlot.colors[self.model_index], 0.5),
                    showlegend=True,
                )
                self.fig.add_trace(upp)

        elif self.ploter == 'matplotlib':
            self.ax.plot(xs, ys, lw=1.0, color=ModelPlot.colors[self.model_index], label=model.expr)
            if post:
                self.ax.fill_between(
                    xs,
                    ys_ci[0],
                    ys_ci[1],
                    fc=ModelPlot.colors[self.model_index],
                    alpha=0.5,
                    label=f'{model.expr} CI',
                )
            self.ax.legend()

        if post:
            self.fig_data[model.expr] = {'xs': xs, 'ys': ys, 'ys_ci': ys_ci}
        else:
            self.fig_data[model.expr] = {'xs': xs, 'ys': ys}

    def get_fig(self):
        """Wrap the accumulated plot in a :class:`Figure` for display or saving."""

        return Figure(self.fig, self.fig_data, self.ploter)


class Figure:
    """Backend-agnostic figure wrapper with notebook auto-display and saving.

    Shows plotly figures immediately when running in an IPython kernel and
    supports saving to HTML, PDF, or JSON depending on the backend.

    Attributes:
        fig: Underlying figure object.
        fig_data: Raw plotted arrays, saved alongside the figure as JSON.
        plotter: Backend tag -- ``'plotly'``, ``'matplotlib'``,
            ``'cornerpy'``, or ``'getdist'``.
    """

    def __init__(self, fig, fig_data, plotter):
        """Store the figure and auto-display it when running in a notebook.

        Args:
            fig: Backend-specific figure object.
            fig_data: Raw plotted arrays, or ``None`` if not exported.
            plotter: Backend tag.
        """

        self.fig = fig
        self.fig_data = fig_data
        self.plotter = plotter

        if self.is_notebook() and self.plotter == 'plotly':
            self.fig.show()

    @staticmethod
    def is_notebook():
        """Return ``True`` when running inside an IPython kernel."""

        return 'ipykernel' in sys.modules

    def save(self, fname):
        """Persist the figure (plus raw data) to disk using ``fname`` as stem.

        The extension is picked per backend: ``.html`` for plotly, ``.pdf``
        for matplotlib and cornerpy, and a getdist-native export otherwise.
        Raw ``fig_data`` is additionally dumped as ``<fname>.json``.

        Args:
            fname: Target file path without extension.

        Raises:
            ValueError: If ``plotter`` is not recognized.
        """

        if self.fig_data is not None:
            json_dump(self.fig_data, f'{fname}.json')

        if self.plotter == 'plotly':
            self.fig.write_html(f'{fname}.html', include_plotlyjs='cdn')
            self.fig.write_image(f'{fname}.pdf')
        elif self.plotter == 'matplotlib' or self.plotter == 'cornerpy':
            self.fig.savefig(f'{fname}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close(self.fig)
        elif self.plotter == 'getdist':
            self.fig.export(f'{fname}.pdf')
        else:
            raise ValueError(f'unsupported plotter: {self.plotter}')
