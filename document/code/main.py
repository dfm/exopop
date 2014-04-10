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
import emcee
import numpy as np
import cPickle as pickle
from itertools import product
import matplotlib.pyplot as pl
import scipy.optimize as op
from scipy.misc import logsumexp

from load_data import (transit_lnprob0, ln_period0, load_completenes_sim,
                       load_candidates)
from population import (CensoringFunction, Histogram,
                        SeparablePopulation,
                        ProbabilisticModel, Dataset, Population,
                        SmoothPopulation)


def main():
    bp = "main"
    try:
        os.makedirs(bp)
    except os.error:
        pass

    # Bins chosen to include EP's bins.
    per_rng, np_bins = np.log([6.25, 400.0]), 4*6
    rp_rng, nr_bins = np.log([0.5, 32]), 4*12

    # Load the data.
    ln_P_inj, ln_R_inj, recovered = load_completenes_sim(per_rng=per_rng,
                                                         rp_rng=rp_rng)
    tlp = lambda lnp, lnr: transit_lnprob0 - 2.*(lnp - ln_period0)/3

    # Set up the censoring function.
    censor = CensoringFunction(np.vstack((ln_P_inj, ln_R_inj)).T, recovered,
                               bins=(np_bins, nr_bins),
                               range=[per_rng, rp_rng],
                               transit_lnprob_function=tlp)

    # The values from EP's paper (+some made up numbers).
    lpb, lrb = censor.bins
    x, y = lpb[::4], lrb[::4]
    p_vals = np.array([8.9, 13.7, 15.8, 15.2, 15., 14.8])
    r_vals = np.array([11, 11.5, 12, 14.2, 18.6, 5.9, 1.9, 1, 0.9, 0.7,
                       0.5, 0.5])
    literature = [(x, p_vals / np.sum(p_vals*np.diff(x))),
                  (y, r_vals / np.sum(r_vals*np.diff(y)))]

    # Load the candidates.
    ids, catalog, err = load_candidates()
    print(catalog)
    print(err)
    print(np.mean(err / catalog, axis=0))
    dataset = Dataset.sample(catalog, err, samples=100, censor=censor,
                             functions=[np.log, np.log])
    print("{0} entries in catalog".format(len(catalog)))

    # Build the binned model.
    bins = [x, y]
    print("Run inference on a grid with shape: {0}"
          .format([len(b)-1 for b in bins]))
    pop = Population(bins, censor.bins, 11.0)
    pop = SmoothPopulation([7.0, 0.2, 0.0, 0.2], pop)
    model = ProbabilisticModel(dataset, pop, censor)

    # Compute the vmax histogram.
    ix0 = np.digitize(np.log(catalog[:, 0]), bins[0])
    iy0 = np.digitize(np.log(catalog[:, 1]), bins[1])
    ix = np.digitize(np.log(catalog[:, 0]), lpb)
    iy = np.digitize(np.log(catalog[:, 1]), lrb)
    lp = censor.lnprob[ix, iy]
    grid = np.zeros((len(bins[0])-1, len(bins[1])-1))
    counts = np.zeros_like(grid)
    for i, j in product(range(len(bins[0])-1), range(len(bins[1])-1)):
        m = (ix0 == i+1) * (iy0 == j+1)
        counts[i, j] = np.sum(m)
        if counts[i, j] == 0:
            continue
        v = lp[m]
        grid[i, j] = np.sum(np.exp(-v[np.isfinite(v)]))
    grid[np.isinf(grid)] = 0.0

    # Turn the vmax numbers into something that we can plot.
    lg = np.log(grid)
    v = pop.initial()
    v[-len(lg.flatten()):] = lg.flatten()
    if pop.evaluate(v) is None:
        print("********** failed")
        return

    # Plot the vmax results.
    fig = pop.plot_2d(v, censor=censor, catalog=np.log(catalog),
                      labels=["$\ln T/\mathrm{days}$", "$\ln R/R_\oplus$"],
                      top_axes=["$T\,[\mathrm{days}]$", "$R\,[R_\oplus]$"],
                      literature=literature, alpha=1)
    fig.savefig(os.path.join(bp, "vmax.png"))
    assert 0


if __name__ == "__main__":
    main()
