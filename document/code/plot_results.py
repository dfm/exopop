#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pl
from scipy.misc import logsumexp

from load_data import load_petigura_bins

ep = load_petigura_bins()
censor, dataset, pop, sampler = pickle.load(open(sys.argv[1]))

samples = sampler.chain
# samples = samples.reshape((-1, samples.shape[-1]))

# # Subsample the chain to get some posterior samples to display.
# subsamples = samples[np.random.randint(len(samples), size=24)]
# lnrates = map(pop.evaluate, subsamples)
# y = logsumexp(lnrates[10] + censor.ln_bin_widths[0][:, None], axis=0)
# y -= logsumexp(y + censor.ln_bin_widths[1])
# x = censor.bins[1]
# pl.plot(x[:-1], np.exp(y))
pl.plot(samples[:, :, 1].T)
pl.savefig("test.png")

# fig_per, fig_rp = pop.plot(,
#                            rp_label=rp_label, ep=ep)
# fig_per.savefig(os.path.splitext(sys.argv[1])[0]+"-period.png")
# fig_rp.savefig(os.path.splitext(sys.argv[1])[0]+"-radius.png")
