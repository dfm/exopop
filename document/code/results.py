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
    i = int(f.attrs["iteration"])
    samples = f["samples"][:i, :]
    hyper = f["hyper"][:i, :]
    lnprob = f["lnprob"][:i]

for i in range(hyper.shape[1]):
    pl.clf()
    pl.plot(hyper[:, i])
    pl.savefig(os.path.join(bp, "time-hyper-{0:03d}.png".format(i)))

samples = samples[-20000:, :][::100, :]
print(np.sqrt(np.diag(np.cov(hyper.T))))
print(np.median(hyper, axis=0))

# Compute and plot gamma_earth.
rates = np.exp(pop.get_lnrate(samples, [np.log(365.), np.log(1.0)]))
fracs = rates / 42557.0
# fracs = rates / 42557.0*(np.log(2) - np.log(1)) * (np.log(400) - np.log(200))
a, b, c = triangle.quantile(fracs, [0.16, 0.5, 0.84])
print("{0}^{{+{1}}}_{{-{2}}}".format(b, c-b, b-a))
pl.hist(fracs, 100, color="k", histtype="step")
pl.savefig(os.path.join(bp, "rate.png"))

# Plot some posterior samples of the rate function.
somesamples = samples[np.random.randint(len(samples), size=50), :]
fig = pop.plot_2d(somesamples, censor=model.censor, catalog=np.log(catalog),
                  err=err, true=truth, labels=labels, top_axes=top_axes,
                  literature=literature)
fig.savefig(os.path.join(bp, "results.png"))
# fig.savefig(os.path.join(bp, "results.pdf"))
