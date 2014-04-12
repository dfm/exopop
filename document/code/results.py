#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

try:
    from savefig import monkey_patch
except ImportError:
    pass
else:
    monkey_patch()

import os
import sys
import h5py
import triangle
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pl

import load_data

bp = sys.argv[1]
model, catalog, err, truth, labels, top_axes, literature = \
    pickle.load(open(os.path.join(bp, "model.pkl")))
pop = model.population

with h5py.File(os.path.join(bp, "results.h5")) as f:
    chain = f["chain"][...]
    lnprob = f["lnprob"][...]

samples = chain[:, -5000:, :].reshape((-1, chain.shape[2]))

rates = np.exp(pop.get_lnrate(samples, [np.log(365.), np.log(1.0)]))
# fracs = rates / 42557.0
fracs = rates / 42557.0*(np.log(2) - np.log(1)) * (np.log(400) - np.log(200))
a, b, c = triangle.quantile(fracs, [0.16, 0.5, 0.84])
print("{0}^{{+{1}}}_{{-{2}}}".format(b, c-b, b-a))
pl.hist(fracs, 100, color="k", histtype="step")
pl.savefig(os.path.join(bp, "rate.png"))
