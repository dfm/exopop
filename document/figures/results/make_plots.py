#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy as np
from triangle import quantile
import matplotlib.pyplot as pl
from scipy.misc import logsumexp
from matplotlib.ticker import FormatStrFormatter

with h5py.File("samples.h5", "r") as f:
    samples = f["ln_occurrence_rate_samples"][...]
    ln_period_bin_edges = f["ln_period_bin_edges"][...]
    ln_radius_bin_edges = f["ln_radius_bin_edges"][...]

bins = [ln_period_bin_edges, ln_radius_bin_edges]
labels = ["R / R_\oplus", "P / \mathrm{day}"]
slices = [
    [slice(0, 4), slice(4, 7), slice(7, 10), slice(10, 13)],
    [slice(0, 3), slice(3, 5), slice(5, 7)]
]
colors = "brgc"
linestyles = ["-", "-", "-", "-"]


def plot_results(a):
    x = bins[a % 2]
    s = [slice(None), slice(None), slice(None)]

    x0 = np.linspace(np.exp(x[0]), np.exp(x[-1]), 5000)[:-1]
    inds = np.digitize(np.log(x0), x) - 1

    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax_right = ax.twinx()

    fig2 = pl.figure()
    ax2 = fig2.add_subplot(111)
    ax_right2 = ax2.twinx()

    for i, (blah, ls, c) in enumerate(zip([slice(None)] + slices[a % 2],
                                          ["-"] + linestyles, "k" + colors)):
        s[a] = blah
        samps = logsumexp(samples[s], axis=a)
        # y = np.mean(samps, axis=0) - np.log(42557.0)
        # yerr = np.std(samps, axis=0)
        q = np.array([quantile(_ - np.log(42557.0), [0.16, 0.5, 0.84])
                      for _ in samps.T])
        y = q[:, 1]
        yerr = np.vstack((q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]))

        if i == 0:
            txt = "all"
        else:
            rng = np.exp(min(bins[a-1][blah])), np.exp(max(bins[a-1][blah]))
            txt = r"({0:.1f}, {1:.1f})".format(*rng)
            print(txt)

        ax_right.plot(np.array(zip(x[:-1], x[1:])).flatten(),
                      np.array(zip(y, y)).flatten(),
                      color=c, ls=ls, lw=0.5*(len(slices[a % 2])-i)+1)
        ax_right.errorbar(0.5*(x[:-1]+x[1:])+(i-0.5*len(slices[a % 2]))*0.03,
                          y, yerr=yerr, fmt="+", color=c, capsize=0)

        # Plot the linear plot.
        y0 = y[inds] - np.log(x0)
        ax_right2.plot(x0, y0, color=c)
        binx = 0.5*(x[:-1]+x[1:])
        ax_right2.errorbar(np.exp(binx)+(i-0.5*len(slices[a % 2]))*0.035,
                           y-binx, yerr=yerr, fmt="+", color=c, capsize=0,
                           ms=0)

    ax_right.set_yticklabels([])
    ax.set_xlim(min(x), max(x))
    ax.set_xlabel(r"$\ln {0}$".format(labels[a-1]))
    ax.set_ylabel(r"$N_\mathrm{avg} / N_\star$")

    a2 = ax.twiny()
    a2.set_xlim(np.exp(ax.get_xlim()))
    a2.set_xscale("log")
    a2.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    a2.set_xlabel(r"${0}$".format(labels[a-1]))

    ax.set_ylim(np.exp(ax_right.get_ylim()))
    ax.set_yscale("log")
    ax.set_ylabel(r"$N_\mathrm{avg} / N_\star$")

    fig.subplots_adjust(bottom=0.15, top=0.85)

    if a == 1:
        ax_right2.set_xlim(0, 8)
        ax_right2.set_ylim(-7, 1)

    ax_right2.set_yticklabels([])
    ax2.set_ylim(np.exp(ax_right2.get_ylim()))
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$N_\mathrm{avg} / N_\star$")
    ax2.set_xlabel(r"${0}$".format(labels[a-1]))
    fig2.subplots_adjust(bottom=0.15, top=0.85)

    return fig, fig2

fig, fig2 = plot_results(1)
fig.savefig("radius.pdf")
fig2.savefig("linear-radius.pdf")

fig, fig2 = plot_results(2)
fig.savefig("period.pdf")
