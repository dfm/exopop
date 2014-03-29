#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import numpy as np
import cPickle as pickle

from load_data import load_petigura_bins

ep = load_petigura_bins()

# Load the MCMC results.
bp = sys.argv[1]
fn = os.path.join(bp, "results.pkl")
censor, dataset, pop, samples, lnprob = pickle.load(open(fn))

# Remove a burn-in and flatten the chain.
if len(sys.argv) > 2:
    burnin = int(sys.argv[2])
else:
    burnin = 4000

# import matplotlib.pyplot as pl
# pl.plot(samples[:, :, 0].T)
# pl.savefig("test.png")
# assert 0
samples = samples[:, burnin:, :]
samples = samples.reshape((-1, samples.shape[-1]))

# Subsample the chain to get some posterior samples to display.
subsamples = samples[np.random.randint(len(samples), size=100)]
figs = pop.plot(subsamples,
                # ranges=[np.log([6.25, 100]), np.log([1.0, 16.0])],
                labels=["$\ln T/\mathrm{days}$", "$\ln R/R_\oplus$"],
                top_axes=["$T\,[\mathrm{days}]$", "$R\,[R_\oplus]$"],
                literature=ep)
figs[0].savefig(os.path.join(bp, "period.png"))
figs[0].savefig(os.path.join(bp, "period.pdf"))
figs[1].savefig(os.path.join(bp, "radius.png"))
figs[1].savefig(os.path.join(bp, "radius.pdf"))
