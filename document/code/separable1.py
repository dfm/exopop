#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import emcee
import numpy as np
import cPickle as pickle
from multiprocessing import Pool

from load_data import (transit_lnprob0, ln_period0, load_completenes_sim,
                       load_candidates, load_petigura_bins)
from population import (CensoringFunction, BrokenPowerLaw, Histogram,
                        SeparablePopulation, NormalizedPopulation,
                        ProbabilisticModel)

# Load the data.
ep = load_petigura_bins()
ln_P_inj, ln_R_inj, recovered = load_completenes_sim()
censor = CensoringFunction(ln_P_inj, ln_R_inj, recovered,
                           nbin_log_per=48, nbin_log_rp=32,
                           log_per_range=(ep[0].min(), ep[0].max()),
                           log_rp_range=(ep[1].min(), ep[1].max()),
                           # log_per_range=np.log([5, 400]),
                           # log_rp_range=np.log([0.5, 16]),
                           transit_lnprob0=transit_lnprob0,
                           ln_period0=ln_period0)
dataset = load_candidates(censor)

# # Plot the completeness function.
# pl.figure(figsize=(10, 6))
# x = censor.log_per_bins
# y = censor.log_rp_bins
# pl.pcolor(x, y, np.exp(censor.lncompleteness).T, cmap="gray")
# pl.plot(dataset.log_per_obs, dataset.log_rp_obs, ".r", ms=3)
# pl.xlim(censor.log_per_bins.min(), censor.log_per_bins.max())
# pl.ylim(censor.log_rp_bins.min(), censor.log_rp_bins.max())
# pl.xlabel(r"$\ln P$")
# pl.ylabel(r"$\ln R_P$")
# pl.colorbar()
# pl.savefig("separable1-completeness.pdf")

# Build the probabilistic model.
log_per_dist = BrokenPowerLaw(censor.log_per_bins)
log_rp_dist = Histogram(censor.log_rp_bins, resample=4)
pop = NormalizedPopulation(11., SeparablePopulation(log_per_dist, log_rp_dist))
model = ProbabilisticModel(dataset, pop)

# Plot the initial distribution.
fig = pop.plot(pop.initial() + 0.1 * np.random.randn(len(pop.initial())),
               ep=ep)
fig.savefig("separable1-initial.pdf")

# Set up the sampler.
p0 = pop.initial()
ndim, nwalkers = len(p0), 32
pos = [p0 + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]

# Make sure that all the initial positions have finite probability.
assert np.all(np.isfinite(map(model.lnprob, pos)))

# Set up the sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, model, pool=Pool())
pos, lp, state = sampler.run_mcmc(pos, 3000)
sampler.reset()
sampler.run_mcmc(pos, 2000)

pickle.dump((censor, dataset, pop, sampler), open("separable1.pkl", "wb"), -1)

samples = sampler.chain
samples = samples.reshape((-1, ndim))

fig = pop.plot(samples[np.random.randint(len(samples), size=24)],
               title="posterior sampling", ep=ep)
fig.savefig("separable1.pdf")
