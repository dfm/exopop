#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script loads the results of a `main.py` run and performs a consistent
analysis generating the main figures from the paper and thinning the MCMC
chain.

Usage:

    python results.py path/to/mcmc/output/

"""

from __future__ import division, print_function

import os
import sys
import h5py
import triangle
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pl

from emcee.autocorr import integrated_time

# HACK: make sure that the import path is set up properly.
import load_data
load_data = load_data

bp = sys.argv[1]
model, catalog, err, truth, labels, top_axes, literature = \
    pickle.load(open(os.path.join(bp, "model.pkl")))
pop = model.population

# Load the original catalog if it's there.
fn = os.path.join(bp, "catalog.pkl")
if os.path.exists(fn):
    catalog, err, truth = \
        pickle.load(open(os.path.join(bp, "catalog.pkl")))

    # Hack to deal with negative radii.
    m = catalog[:, 1] > 0
    catalog = catalog[m]
    rerr = [np.log(catalog[:, 1]) - np.log(np.abs(catalog[:, 1]-err[m, 1])),
            np.log(catalog[:, 1]+err[m, 1]) - np.log(catalog[:, 1])]
    err = [0, rerr]

labels = ["$\ln P/\mathrm{day}$", "$\ln R/R_\oplus$"]
top_axes = ["$P\,[\mathrm{days}]$", "$R\,[R_\oplus]$"]


with h5py.File(os.path.join(bp, "results.h5")) as f:
    i = int(f.attrs["iteration"])
    samples = f["samples"][:i, :]
    hyper = f["hyper"][:i, :]
    lnprob = f["lnprob"][:i]

for i in range(hyper.shape[1]):
    pl.clf()
    pl.plot(hyper[:, i])
    pl.savefig(os.path.join(bp, "time-hyper-{0:03d}.png".format(i)))

pl.clf()
pl.plot(lnprob)
pl.savefig(os.path.join(bp, "time-lnprob.png"))

nstar = 42557.0
ntot = 200000
samples = samples[-ntot:, :]  # [::50, :]

# Reformat the samples and save the samples.
thin_by = int(np.min(integrated_time(samples, axis=0)))
thinned = samples[::thin_by, :]
grids = thinned.reshape((len(thinned), pop.shape[0], pop.shape[1]))
print(grids.shape)
print([b.shape for b in pop.bins])


def xmap(f, i):
    return (f(*x) for x in i)

h_mu = np.mean(np.exp(0.5 * hyper[-ntot:][::thin_by, 2:]), axis=0)
h_std = np.std(np.exp(0.5 * hyper[-ntot:][::thin_by, 2:]), axis=0)
print("Hyper:")
print("\n".join(xmap("{0} ± {1}".format, zip(h_mu, h_std))))
print()

with h5py.File(os.path.join(bp, "samples.h5"), "w") as f:
    f.create_dataset("hyperparameter_samples", data=hyper[-ntot:][::thin_by])
    f.create_dataset("ln_occurrence_rate_samples", data=grids)
    f.create_dataset("ln_period_bin_edges", data=pop.bins[0])
    f.create_dataset("ln_radius_bin_edges", data=pop.bins[1])

# Load the true gamma earth if it exists.
fn = os.path.join(bp, "gamma.txt")
if os.path.exists(fn):
    gamma = [np.exp(float(open(fn).read())) / nstar]
else:
    gamma = None  # [5.7 / 100, 1.7 / 100, 2.2 / 100]

# Load the extrapolated value.
factor = (np.log(400) - np.log(200)) * (np.log(2) - np.log(1))
if os.path.exists(fn):
    ext = np.array(open(os.path.join(bp, "extrap.txt"), "r").read().split(),
                   dtype=float) / 42557.0
else:
    ext = np.array([0.057, -0.022, 0.017])

ext /= factor
print(ext)

# Compute and plot gamma_earth.
rates = pop.get_lnrate(samples, [np.log(365.), np.log(1.0)])
fracs = rates - np.log(42557.0)
a, b, c = triangle.quantile(fracs, [0.16, 0.5, 0.84])
print("{0}^{{+{1}}}_{{-{2}}}".format(b, c-b, b-a))
al, bl, cl = triangle.quantile(np.exp(fracs), [0.16, 0.5, 0.84])
print("{0}^{{+{1}}}_{{-{2}}}".format(bl, cl-bl, bl-al))

fig = pl.figure()
ax = fig.add_subplot(111)
ax.hist(fracs, 50, color="k", histtype="step", normed=True)

if gamma is not None:
    ax.axvline(np.log(gamma[0]), color="k", alpha=0.5, lw=3)

ax.axvline(np.log(ext[0]+ext[1]), color="k", ls="dashed")
ax.axvline(np.log(ext[0]+ext[2]), color="k", ls="dashed")
ax.axvline(np.log(ext[0]), color="k")

# ax.axvline(b, color="k")
# ax.axvline(c, color="k", ls="dashed")
# ax.axvline(a, color="k", ls="dashed")

ax.set_xlim(-7, -1)
ax.set_xlabel(r"$\ln \Gamma_\oplus$")
ax.set_ylabel(r"$p(\ln \Gamma_\oplus)$")

a2 = ax.twiny()
a2.set_xlim(np.exp(ax.get_xlim()))
a2.set_xscale("log")
a2.set_xlabel(r"$\Gamma_\oplus$")

fig.subplots_adjust(top=0.85)

fig.savefig(os.path.join(bp, "rate.png"))
fig.savefig(os.path.join(bp, "rate.pdf"))

# Plot some posterior samples of the rate function.
somesamples = samples[np.random.randint(len(samples), size=50), :]
fig = pop.plot_2d(somesamples, censor=model.censor, catalog=np.log(catalog),
                  err=err, true=truth, labels=labels, top_axes=top_axes,
                  literature=literature, norm=nstar, vmin=0, vmax=None)
fig.savefig(os.path.join(bp, "results.png"), dpi=300)
fig.savefig(os.path.join(bp, "results.pdf"))
