#!/usr/bin/env python
# -*- coding: utf-8 -*-

# plasm.py
# Jim Bagrow
# Last Modified: 2021-05-06

"""PLASM - PLot Analysis Spreads for Meetings

A rough-and-ready tool for preparing all combinations of log- and linear-scaled
axes for a scatterplot or distribution plot (CDF, CCDF, or histogram), to save
for reference.

PLASM consists of two functions:

    CHASM - Cdf and Histogram Analysis Spread for Meetings
    SPASM - ScatterPlot Analysis Spread for Meetings.

Now when you show someone a scatter plot and they ask for a log-scaled x-axis,
you can pull out the saved SPASM and have every combination of log- and
linear-scaled axes ready to go.
"""

import getpass
from datetime import datetime
import numpy as np
import scipy, scipy.stats
import matplotlib.pyplot as plt


def _aver_xy(X,Y):
    """return new X,Y where X is each unique value of X and Y is the average of
    all the y-values corresponding to that X.
    """
    X = np.array(X)
    Y = np.array(Y)
    means = {}
    stdvs = {}
    for xi in np.unique(X):
        yi = Y[np.where(X == xi)]
        means[xi] = np.mean( yi )
        stdvs[xi] = np.std(yi)
    x    = np.array(sorted(means.keys()))
    y    = np.array([ means[xi] for xi in x ])
    ebar = np.array([ stdvs[xi] for xi in x ])
    return x,y,ebar


def _get_ccdf(data):
    """Compute the complementary (empirical) cumulative distribution function.
    Do it the right way, by sorted the data points and not binning.

    Example:
    >>> data = np.random.randn(1000,1)
    >>> x,Cx = blt.ccdf(data)
    >>> plt.plot(x, Cx)
    >>> plt.xlabel("x")
    >>> plt.ylabel("Prob(X > x)")
    """
    data_s = sorted(data)
    return data_s, 1.0 - 1.0 * np.arange(len(data_s)) / len(data_s)


def _logize_axes(axes, title=False):
    """axes is a 1x4 list of ax handles."""
    axes[1].set_xscale('log')
    axes[2].set_yscale('log')
    axes[3].set_xscale('log')
    axes[3].set_yscale('log')
    if title:
        axes[0].set_title("Linear",    fontsize='medium')
        axes[1].set_title("Semilog X", fontsize='medium')
        axes[2].set_title("Semilog Y", fontsize='medium')
        axes[3].set_title("Log-Log",   fontsize='medium')


def _sign_text():
    u = getpass.getuser().upper()
    nt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Created by: {u}\nLast modified: {nt}"


def _finalize_figure(filename=None):
    plt.tight_layout(w_pad=0.25)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def _autobins(data, log=False):
    list_mins = []
    list_maxs = []
    list_nums = []
    for dat in data:
        if log:
            dat = np.log(dat[dat>0])
        list_mins.append(min(dat))
        list_maxs.append(max(dat))
        _, bin_edges = np.histogram(dat, bins='auto')
        list_nums.append(len(bin_edges)-1)
    nb = int(np.mean(list_nums))
    if log:
        nb //= 2

    r = (min(list_mins), max(list_maxs))
    bin_edges = np.histogram_bin_edges([1], bins=nb, range=r)
    return bin_edges, nb


def chasm(*data, names='x', filename=None, show_median=False, show_stats=True):
    """
    CHASM: Cdf and Histogram Analysis Spread for Meetings.

    Generate a set of plots showing all the variations of log-scaled axes for
    the CDF, CCDF, and (a binned) histogram.  Summary statistics and metadata
    are printed on the right.

    Histograms are not yet implemented.

    Args:
        *data : list, or list of lists, of numeric values.
        names : names of variable(s), to be used in summary stats.
        filename : name of file to save plot if not None.
        show_median : draw horizontal line on CDFs.
        show_stats : Show summary statistics in right column.

    Returns:
        Figure and axes handles for the created plot.

    Example:
        >>> X1 = np.random.randn(1000,)
        >>> X2 = np.random.randn(1000,)+0.4
        >>>
        >>> names = ["dat1","dat2"]
        >>> chasm(X1, X2, names=names, show_median=True)
    """

    fig,axes = plt.subplots(3,5, figsize=(8.5*1.67,3.5*2))

    # get and plot (C)CDF(s)
    for dat in data:
        x,Cx = _get_ccdf(dat)
        for ax in axes[0][:-1]:
            ax.plot(x,Cx) # use plt.step?
        for ax in axes[1][:-1]:
            ax.plot(x,1-Cx) # use plt.step?
    if show_median:
        for ax in axes[:2,:-1].flatten():
            ax.axhline(0.5, ls='--', lw=0.75, color='k')

    # get and plot histogram(s)
    bin_edges_lin, num_bins_lin = _autobins(data)
    bin_edges_log, num_bins_log = _autobins(data, log=True)
    bin_edges_log = np.exp(bin_edges_log)
    alpha = 1 if len(data) == 1 else 0.33
    for i,dat in enumerate(data):
        cs = f'C{i}'
        for c,ax in enumerate(axes[2][:-1]):
            bin_edges = bin_edges_lin
            if c in [1,3]:
                bin_edges = bin_edges_log
            ax.hist(dat, bins=bin_edges, alpha=alpha, histtype='stepfilled', density=True, color=cs)
            ax.hist(dat, bins=bin_edges, alpha=1.0,   histtype='step',       density=True, lw=1., color=cs)


    # label and log-ize the plots:
    for ax in axes[2]:
        ax.set_xlabel("$x$")
        ax.set_ylabel("Prob. density", labelpad=2)
    for ax in axes[1]:
        ax.set_ylabel("Pr$(X < x)$", labelpad=2)
    for ax in axes[0]:
        ax.set_ylabel("Pr$(X > x)$", labelpad=2)

    _logize_axes(axes[0,:], title=True)
    _logize_axes(axes[1,:])
    _logize_axes(axes[2,:])

    # add summary stats in fifth panel:
    if show_stats:
        for i,dat in enumerate(data):
            prcL = np.percentile(dat,  2.5)
            prcR = np.percentile(dat, 97.5)
            lbl = ""
            if names != "x" and len(names) == len(data):
                lbl += f"$x =${names[i]}:\n"
            lbl += f"mean = {np.mean(dat):0.6f}\n"
            lbl += f"stdv = {np.std(dat):0.6f}\n"
            lbl += f"median = {np.median(dat):0.6f}\n"
            lbl += f"    95 = {prcL:0.4f}, {prcR:0.4f}"
            fig.text(0.82,0.9-i*0.15,lbl,
                     fontfamily='monospace', va='top', color=f'C{i}')
        fig.text(0.82, 0.150, f"Hist. lin. bins: $n={num_bins_lin}$",
                 fontfamily='monospace', va='top')
        fig.text(0.82, 0.125, f"Hist. log. bins: $n={num_bins_log}$",
                 fontfamily='monospace', va='top')
        axes[0,4].set_title("Summary statistics", fontsize='medium', ha='right')
    axes[0,4].axis('off')
    axes[1,4].axis('off')
    axes[2,4].axis('off')

    # sign and finalize the figure:
    fig.text(0.818,0.08, _sign_text(), fontsize='small', va='top')
    _finalize_figure(filename=filename)
    return fig, axes


def spasm(*data, xlabel=r'$X$', ylabel=r'$Y$', names=None,
          show_trend=False, show_line=False, filename=None,
          show_stats=True, **kwargs):
    """SPASM: ScatterPlot Analysis Spread for Meetings.

    Generate a set of scatter plots showing all the variations of log-scaled
    axes. Summary statistics and metadata are printed on the right.

    Args:
        *data : list, or list of lists, of numeric values.
        xlabel,ylabel : names of variable(s), to be used in axis labels.
        names : NOT IMPLEMENTED YET.
        show_trend : Show binned trend (NOT YET IMPLEMENTED).
        show_line : Show simple linear regression (NOT YET IMPLEMENTED).
        filename : name of file to save plot if not None.
        show_stats : Show summary statistics in right column.
        **kwargs : Additional keywords will be passed to plot command.

    Returns:
        Figure and axes handles for the created plot.

    Example:
        >>> n = 150
        >>> X1 = np.random.randn(n,)
        >>> e  = np.random.randn(n,)*0.1
        >>> Y1 = 0.3 + 1.2 * X1 + e

        >>> X2 = -1 + 2*np.random.random(n,)
        >>> Y2 = 2*X2**2 + e

        >>> spasm(X1,Y1, X2,Y2, names=['Expr-1', 'Expr-2'])
    """
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.5
        kwargs['mec'] = 'none'

    # if data is length one assume, it's x and y put together
    if len(data) == 1:
        X,Y = data[0] # what about a numpy matrix?
        Xs,Ys = [X],[Y]
    else:
        Xs = data[0::2]
        Ys = data[1::2]

    fig,axes = plt.subplots(1,5, figsize=(8.5*1.67,3.))


    # plot the data:
    for X,Y in zip(Xs,Ys):
        for ax in axes[:-1]:
            ax.plot(X,Y, 'o', **kwargs)

    # label and log-ize the plots:
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, labelpad=1)

    _logize_axes(axes, title=True)


    # add summary stats in fifth panel:
    if show_stats:
        if names is None:
            names = [f"Data {i+1}" for i in range(len(Xs))]

        for i,(x,y) in enumerate(zip(Xs,Ys)):
            pearR,pearP = scipy.stats.pearsonr(x,y)
            speaR,speaP = scipy.stats.spearmanr(x,y)
            lbl = ""
            lbl += f"{names[i]}:\n"
            lbl += f"Pearson $r =${pearR:0.3f} ($p =${pearP:0.3f})\n"
            lbl += f"Spearman $\\rho =${speaR:0.3f} ($p =${speaP:0.3f})\n"
            fig.text(0.818,0.88-i*0.2,lbl, fontsize='small',
                     fontfamily='monospace', va='top', color=f'C{i}')
        axes[4].set_title("Summary statistics", fontsize='medium', ha='right')
    axes[4].axis('off')
    axes[4].axis('off')

    # sign and finalize the figure:
    fig.text(0.818,0.15, _sign_text(), fontsize='small', va='top')
    _finalize_figure(filename=filename)
    return fig, axes


if __name__ == '__main__':

    n = 1000
    X = np.random.randn(n,)
    Y = np.random.randn(n,)+0.4
    Z = np.random.randn(n,)+0.8
    #W = 1+np.random.pareto(3, n)

    names = ["data1","data2","data3"]
    chasm(X, Y, Z, names=names, show_median=True, filename='figures/example-chasm.png')


    n = 150
    X1 = np.random.randn(n,)
    e  = np.random.randn(n,)*0.1
    Y1 = 0.3 + 1.2 * X1 + e

    X2 = -1 + 2*np.random.rand(n,)
    Y2 = 2*X2**2 + e

    spasm(X1,Y1, X2,Y2, names=['Expr-1', 'Expr-2'],
          filename='figures/example-spasm.png')
