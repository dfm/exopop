#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import emcee
import numpy as np
import cPickle as pickle
from itertools import product
import matplotlib.pyplot as pl

from load_data import (transit_lnprob0, ln_period0, load_completenes_sim,
                       load_candidates, load_petigura_bins)
from population import (CensoringFunction, BrokenPowerLaw, Histogram,
                        SeparablePopulation, NormalizedPopulation,
                        ProbabilisticModel, Dataset)

try:
    os.makedirs("fake")
except os.error:
    pass

# Define the box that we're going to work in.
per_rng = np.log([5.0, 400.0])
rp_rng = np.log([0.5, 64.0])

# Load the data.
ln_P_inj, ln_R_inj, recovered = load_completenes_sim(per_rng=per_rng,
                                                     rp_rng=rp_rng)
tlp = lambda lnp, lnr: transit_lnprob0 - 2.*(lnp - ln_period0)/3

# Set up the censoring function.
censor = CensoringFunction(np.vstack((ln_P_inj, ln_R_inj)).T, recovered,
                           bins=(32, 36),
                           range=[per_rng, rp_rng],
                           transit_lnprob_function=tlp)

# Build a synthetic population.
lpb, lrb = censor.bins
pdist = BrokenPowerLaw(lpb)
rdist = BrokenPowerLaw(lrb)
pop0 = SeparablePopulation([pdist, rdist])
pop0 = NormalizedPopulation(10.8, pop0)

# Plot the true distributions.
truth = [10.8, 0.5, -0.2, 4.0, 0.8, -1.5, 1.0]
figs = pop0.plot(truth, alpha=1,
                 labels=["$\ln T/\mathrm{days}$", "$\ln R/R_\oplus$"],
                 top_axes=["$T\,[\mathrm{days}]$", "$R\,[R_\oplus]$"])
figs[0].savefig(os.path.join("fake", "true-period.png"))
figs[1].savefig(os.path.join("fake", "true-radius.png"))

# Sample from this censored population.
lnrate = np.array(censor.lnprob[1:-1, 1:-1])
lnrate += pop0.evaluate(truth)

catalog = np.empty((0, 2))
for i, j in product(xrange(len(lpb)-1), xrange(len(lrb)-1)):
    area = (lpb[i+1] - lpb[i]) * (lrb[i+1] - lrb[i])
    k = np.random.poisson(np.exp(lnrate[i, j]) * area)
    if k == 0:
        continue
    entry = np.vstack((np.random.uniform(lpb[i], lpb[i+1], k),
                       np.random.uniform(lrb[j], lrb[j+1], k))).T
    catalog = np.concatenate((catalog, entry), axis=0)
dataset = Dataset([catalog])
print("{0} entries in catalog".format(len(catalog)))

# Plot the actual rate function.
pl.figure()
pl.pcolor(lpb, lrb, np.exp(censor.lncompleteness[1:-1, 1:-1].T), cmap="gray")
# pl.pcolor(lpb, lrb, np.exp(lnrate.T), cmap="gray")
pl.plot(catalog[:, 0], catalog[:, 1], ".r", ms=3)
pl.colorbar()
pl.xlim(min(lpb), max(lpb))
pl.ylim(min(lrb), max(lrb))
pl.savefig("fake/true-rate.png")

# Run inference with the true population.
model = ProbabilisticModel(dataset, pop0, censor)
print("Initial ln-prob = {0}".format(model.lnprob(pop0.initial())))

# Set up the sampler.
p0 = pop0.initial()
ndim, nwalkers = len(p0), 32
pos = [p0 + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]

# Make sure that all the initial positions have finite probability.
finite = np.isfinite(map(model.lnprob, pos))
assert np.all(finite), "{0}".format(np.sum(finite))

# Set up the sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, model)
sampler.run_mcmc(pos, 5000)

pickle.dump((censor, dataset, pop0, sampler.chain, sampler.lnprobability),
            open("fake/results.pkl", "wb"), -1)
