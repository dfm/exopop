#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import emcee
import numpy as np
import cPickle as pickle

from load_data import (transit_lnprob0, ln_period0, load_completenes_sim,
                       load_candidates, load_petigura_bins)
from population import (CensoringFunction, BrokenPowerLaw, Histogram,
                        SeparablePopulation, NormalizedPopulation,
                        ProbabilisticModel)

try:
    os.makedirs("separable")
except os.error:
    pass

# Define the box that we're going to work in.
per_rng = np.log([5.0, 400.0])
rp_rng = np.log([0.5, 64.0])

# Load the data.
ep = load_petigura_bins()
ln_P_inj, ln_R_inj, recovered = load_completenes_sim(per_rng=per_rng,
                                                     rp_rng=rp_rng)
tlp = lambda lnp, lnr: transit_lnprob0 - 2.*(lnp - ln_period0)/3

# Set up the censoring function.
censor = CensoringFunction(np.vstack((ln_P_inj, ln_R_inj)).T, recovered,
                           bins=(32, 70),
                           range=[per_rng, rp_rng],
                           transit_lnprob_function=tlp)

# Load the dataset and resample under the censoring function.
dataset = load_candidates(censor, 120)

# Define the population.
lpb, lrb = censor.bins
b = np.concatenate(([lrb[0]], ep[1], [lrb[-1]]))
pdist = BrokenPowerLaw(lpb)
rdist = Histogram(b, lrb)
print(rdist.inds)
pop = NormalizedPopulation(11., SeparablePopulation([pdist, rdist]))

# Define the probabilistic model.
model = ProbabilisticModel(dataset, pop, censor)
print("Initial ln-prob = {0}".format(model.lnprob(pop.initial())))

if True:
    import matplotlib.pyplot as pl
    # Plot the completeness function.
    pl.figure(figsize=(10, 6))
    x = censor.bins[0]
    y = censor.bins[1]
    pl.pcolor(x, y, np.exp(censor.lncompleteness[1:-1, 1:-1]).T, cmap="gray")
    pl.plot(dataset.catalogs[:, :, 0], dataset.catalogs[:, :, 1], ".r", ms=3,
            alpha=0.5)
    pl.xlim(x.min(), x.max())
    pl.ylim(y.min(), y.max())
    pl.xlabel(r"$\ln P$")
    pl.ylabel(r"$\ln R_P$")
    pl.colorbar()
    pl.savefig("separable/completeness.png")

# Set up the sampler.
p0 = pop.initial()
ndim, nwalkers = len(p0), 32
pos = [p0 + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]

# Make sure that all the initial positions have finite probability.
finite = np.isfinite(map(model.lnprob, pos))
assert np.all(finite), "{0}".format(np.sum(finite))

# Set up the sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, model)
sampler.run_mcmc(pos, 5000)

pickle.dump((censor, dataset, pop, sampler),
            open("separable/results.pkl", "wb"), -1)
