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
mylog = lambda x: x
ep = load_petigura_bins(mylog=mylog)
ln_P_inj, R_inj, recovered = load_completenes_sim(rp_func=mylog)
censor = CensoringFunction(ln_P_inj, R_inj, recovered,
                           nbin_log_per=48, nbin_log_rp=48,
                           log_per_range=np.log([5, 400]),
                           log_rp_range=mylog([0.5, 16]),
                           transit_lnprob0=transit_lnprob0,
                           ln_period0=ln_period0)
dataset = load_candidates(censor, rp_func=mylog)

# Build the probabilistic model.
log_per_dist = BrokenPowerLaw(censor.log_per_bins)
log_rp_dist = Histogram(censor.log_rp_bins, resample=4)
pop = NormalizedPopulation(11., SeparablePopulation(log_per_dist, log_rp_dist))
model = ProbabilisticModel(dataset, pop)

# Set up the sampler.
p0 = pop.initial()
ndim, nwalkers = len(p0), 32
pos = [p0 + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]

# Make sure that all the initial positions have finite probability.
assert np.all(np.isfinite(map(model.lnprob, pos)))

# Set up the sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, model)
pos, lp, state = sampler.run_mcmc(pos, 3000)
sampler.reset()
sampler.run_mcmc(pos, 2000)

pickle.dump((censor, dataset, pop, sampler), open("separable-linear.pkl", "wb"), -1)
