#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy as np
from triangle import quantile
import matplotlib.pyplot as pl
from scipy.misc import logsumexp
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

with h5py.File("samples.h5", "r") as f:
    samples = f["ln_occurrence_rate_samples"][...]
    ln_period_bin_edges = f["ln_period_bin_edges"][...]
    ln_radius_bin_edges = f["ln_radius_bin_edges"][...]

# Integrate the rate over EP's bin.
ep_inds = (slice(None), 5, slice(2, 4))
ep_area = (np.diff(ln_period_bin_edges)[ep_inds[1]]
           * np.diff(ln_radius_bin_edges)[ep_inds[2]])
factor = np.diff(ln_radius_bin_edges)[ep_inds[2]] * 0.05061 * 10 ** (2./3)
factor *= -3.0 * np.diff(ln_period_bin_edges ** (-2./3))[ep_inds[1]] / 2

rate = np.sum(ep_area * np.exp(samples[ep_inds] - np.log(42557.0)), axis=1)
q = quantile(rate, [0.16, 0.5, 0.84])
print("Integrated rate: {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
      .format(q[1], *(np.diff(q))))

num = np.sum(factor * np.exp(samples[ep_inds]), axis=1)
q = quantile(num, [0.16, 0.5, 0.84])
print("Integrated observable number: {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
      .format(q[1], *(np.diff(q))))

assert 0

bins = [ln_period_bin_edges, ln_radius_bin_edges]
labels = ["R / R_\oplus", "P / \mathrm{day}"]
slices = [
    [slice(0, 4), slice(4, 8), slice(8, 12)],
    [slice(0, 2), slice(2, 4), slice(4, 6)]
]
colors = ["#222222", "#222222", "#222222", "#222222"]
# colors = "brgc"
linestyles = ["--", ":", "-", "-."]
linewidths = [0.75, 0.75, 0.75, 0.75]


def plot_results(a):
    x = bins[a % 2]
    s = [slice(None), slice(None), slice(None)]
    s2 = [None, None, None]

    x0 = np.linspace(np.exp(x[0]), np.exp(x[-1]), 5000)[:-1]
    inds = np.digitize(np.log(x0), x) - 1

    fig = pl.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax_right = ax.twinx()

    fig2 = pl.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(111)
    ax_right2 = ax2.twinx()

    for i, (blah, ls, lw, c) in enumerate(zip([slice(None)] + slices[a % 2],
                                              ["-"] + linestyles,
                                              [1.25] + linewidths,
                                              ["#222222"] + colors)):
        s[a] = blah
        s2[a] = blah
        samps = logsumexp(samples[s] + np.log(np.diff(bins[a - 1]))[s2],
                          axis=a)
        q = np.array([quantile(_ - np.log(42557.0), [0.16, 0.5, 0.84])
                      for _ in samps.T])
        y = q[:, 1]
        yerr = np.vstack((q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]))

        if i == 0:
            txt = "all"
        else:
            txt = []
            ix = range(len(bins[a-1]))[blah]
            print(ix)
            rng = (np.exp(bins[a-1][min(ix)]),
                   np.exp(bins[a-1][max(ix)+1]))
            for r in rng:
                ir = round(r)
                f = 0
                while abs(ir - r) > 10**(-(f+1)) and f < 2:
                    f += 1
                    ir = round(r*10**f) / 10**f
                txt.append("{{0:.{0}f}}".format(f).format(r))
            txt = "${1} \le {0} < {2}$".format(labels[a % 2], *txt)

        ax_right.plot(np.array(zip(x[:-1], x[1:])).flatten(),
                      np.array(zip(y, y)).flatten(),
                      color=c, ls=ls, lw=lw, label=txt)
        ax_right.errorbar(0.5*(x[:-1]+x[1:])+(i-0.5*len(slices[a % 2]))*0.03,
                          y, yerr=yerr, fmt="+", ms=0, color=c, capsize=0,
                          elinewidth=0.5)

        # Plot the linear plot.
        y0 = y[inds] - np.log(x0)
        binx = 0.5*(x[:-1]+x[1:])
        ax_right2.plot(x0, y0, color=c, lw=lw, label=txt, ls=ls)
        ax_right2.errorbar(np.exp(binx)+(i-0.5*len(slices[a % 2]))*0.04,
                           y-binx, yerr=yerr, fmt="+", color=c, capsize=0,
                           ms=0, elinewidth=0.5)

    ax_right.set_yticklabels([])
    ax.set_xlim(min(x), max(x))
    ax.set_xlabel(r"$\ln {0}$".format(labels[a-1]))
    ax.set_ylabel(r"$\Gamma (\ln {0})$".format(labels[a-1]))
    ax.xaxis.set_major_locator(MaxNLocator(5))

    a2 = ax.twiny()
    a2.set_xlim(np.exp(ax.get_xlim()))
    a2.set_xscale("log")
    a2.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    a2.set_xlabel(r"${0}$".format(labels[a-1]))

    ax.set_ylim(np.exp(ax_right.get_ylim()))
    ax.set_yscale("log")

    fig.subplots_adjust(bottom=0.17, top=0.83, right=0.7, left=0.17)
    prop = FontProperties()
    prop.set_size("10")
    ax_right.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop=prop)

    if a == 1:
        ax_right2.set_xlim(0, 16)
        ax_right2.set_ylim(-8.9, 0.2)

    ax_right2.set_yticklabels([])
    ax2.set_ylim(np.exp(ax_right2.get_ylim()))
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\Gamma ({0})$".format(labels[a-1]))
    ax2.set_xlabel(r"${0}$".format(labels[a-1]))
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    fig2.subplots_adjust(bottom=0.17, top=0.83, right=0.7, left=0.17)
    ax_right2.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop=prop)

    return fig, fig2

fig, fig2 = plot_results(1)
fig.savefig("radius.pdf")
fig2.savefig("linear-radius.pdf")

fig, fig2 = plot_results(2)
fig.savefig("period.pdf")
