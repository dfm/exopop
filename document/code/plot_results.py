#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys
import numpy as np
import cPickle as pickle

from load_data import load_petigura_bins

ep = load_petigura_bins()
censor, dataset, pop, sampler = pickle.load(open(sys.argv[1]))

burnin = 4000
samples = sampler.chain[:, burnin:, :]
samples = samples.reshape((-1, samples.shape[-1]))

# Subsample the chain to get some posterior samples to display.
subsamples = samples[np.random.randint(len(samples), size=100)]
figs = pop.plot(subsamples,
                ranges=[np.log([6.25, 100]), np.log([1.0, 16.0])],
                labels=["$\ln T/\mathrm{days}$", "$\ln R/R_\oplus$"],
                top_axes=["$T\,[\mathrm{days}]$", "$R\,[R_\oplus]$"],
                literature=ep)
figs[0].savefig("period.png")
figs[0].savefig("period.pdf")
figs[1].savefig("radius.png")
figs[1].savefig("radius.pdf")
