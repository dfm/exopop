#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import emcee
import numpy as np
import cPickle as pickle

from load_data import (transit_lnprob0, ln_period0, load_completenes_sim,
                       load_candidates, load_petigura_bins)
from population import (CensoringFunction, BrokenPowerLaw, Histogram,
                        SeparablePopulation, NormalizedPopulation,
                        ProbabilisticModel)

# Load the data.
ep = load_petigura_bins()
ln_P_inj, ln_R_inj, recovered = load_completenes_sim()

per_rng = np.log([5.0, 400.0])
rp_rng = np.log([16.0, 64.0])

K = 50000
ln_P_inj = np.append(ln_P_inj, per_rng[0] +
                     (per_rng[1] - per_rng[0]) * np.random.rand(K))
ln_R_inj = np.append(ln_R_inj, rp_rng[0] +
                     (rp_rng[1]-rp_rng[0]) * np.random.rand(K))
recovered = np.append(recovered, np.ones(K, dtype=bool))

censor = CensoringFunction(ln_P_inj, ln_R_inj, recovered,
                           nbin_log_per=48, nbin_log_rp=70,
                           log_per_range=np.log([5, 400]),
                           log_rp_range=np.log([0.5, 64]),
                           # log_per_range=(ep[0].min(), ep[0].max()),
                           # log_rp_range=(ep[1].min(), 2 * ep[1].max()),
                           transit_lnprob0=transit_lnprob0,
                           ln_period0=ln_period0)
dataset = load_candidates(censor, samples=64)

if True:
    import matplotlib.pyplot as pl
    # Plot the completeness function.
    pl.figure(figsize=(10, 6))
    x = censor.log_per_bins
    y = censor.log_rp_bins
    pl.pcolor(x, y, np.exp(censor.lncompleteness).T, cmap="gray")
    pl.plot(dataset.log_per_obs, dataset.log_rp_obs, ".r", ms=3)
    pl.xlim(censor.log_per_bins.min(), censor.log_per_bins.max())
    pl.ylim(censor.log_rp_bins.min(), censor.log_rp_bins.max())
    pl.xlabel(r"$\ln P$")
    pl.ylabel(r"$\ln R_P$")
    pl.colorbar()
    pl.savefig("separable3-completeness.png")

# Build the probabilistic model.
log_per_dist = BrokenPowerLaw(censor.log_per_bins)
b = np.concatenate(([np.log(0.5)], ep[1], [np.log(64.0)]))
log_rp_dist = Histogram(b, censor.log_rp_bins)
print(log_rp_dist.inds)
pop = NormalizedPopulation(11., SeparablePopulation(log_per_dist, log_rp_dist))
model = ProbabilisticModel(dataset, pop)

# Set up the sampler.
p0 = pop.initial()
ndim, nwalkers = len(p0), 32
pos = [p0 + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]

# Make sure that all the initial positions have finite probability.
finite = np.isfinite(map(model.lnprob, pos))
assert np.all(finite), "{0}".format(np.sum(finite))

# Set up the sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, model)
pos, lp, state = sampler.run_mcmc(pos, 3000)
sampler.reset()
sampler.run_mcmc(pos, 2000)

pickle.dump((censor, dataset, pop, sampler), open("separable3.pkl", "wb"), -1)
