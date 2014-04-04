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

from load_data import transit_lnprob0, ln_period0, load_completenes_sim
from population import (CensoringFunction, Histogram,
                        SeparablePopulation,
                        ProbabilisticModel, Dataset, Population,
                        SmoothPopulation)


def main(args, state=None):
    bp, seed = args
    if seed is not None:
        np.random.seed(seed)
    try:
        os.makedirs(bp)
    except os.error:
        pass

    if state is not None:
        np.random.set_state(state)

    # Save the random state.
    pickle.dump(np.random.get_state(),
                open(os.path.join(bp, "state.pkl"), "wb"), -1)

    # Define the box that we're going to work in.
    # per_rng = np.log([6.25, 100.0])
    # rp_rng = np.log([1.0, 16.0])
    # per_rng = np.log([5.0, 400.0])
    # rp_rng = np.log([0.5, 64.0])

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

    # print(np.exp(censor.bins[0][::4]))
    # print(np.exp(censor.bins[1][::4]))

    # The values from EP's paper (+some made up numbers).
    lpb, lrb = censor.bins
    x, y = lpb[::4], lrb[::4]
    p_vals = np.log(np.array([8.9, 13.7, 15.8, 15.2, 15., 14.8]))
    r_vals = np.log(np.array([11, 11.5, 12, 14.2, 18.6, 5.9, 1.9, 1, 0.9, 0.7,
                              0.5, 0.5]))

    # Normalize the underlying distribution.
    p_vals -= logsumexp(p_vals + np.log(np.diff(x)))
    r_vals -= logsumexp(r_vals + np.log(np.diff(y)))

    # Build a synthetic population.
    truth = np.concatenate([[11.2], p_vals, r_vals])
    pdist = Histogram(x, base=lpb)
    rdist = Histogram(y, base=lrb)
    pop0 = SeparablePopulation([pdist, rdist], lnnorm=truth[0])

    # Plot the true distributions.
    literature = [(pdist.base, np.exp(pdist(truth[1:1+len(p_vals)]))),
                  (rdist.base, np.exp(rdist(truth[1+len(p_vals):])))]
    figs = pop0.plot(truth,
                     labels=["$\ln T/\mathrm{days}$", "$\ln R/R_\oplus$"],
                     top_axes=["$T\,[\mathrm{days}]$", "$R\,[R_\oplus]$"],
                     literature=literature)
    figs[0].savefig(os.path.join(bp, "true-period.png"))
    figs[0].savefig(os.path.join(bp, "true-period.pdf"))
    figs[1].savefig(os.path.join(bp, "true-radius.png"))
    figs[1].savefig(os.path.join(bp, "true-radius.pdf"))

    # Sample from this censored population.
    lnrate = np.array(censor.lnprob[1:-1, 1:-1])
    lnrate += pop0.evaluate(truth)

    catalog = np.empty((0, 2))
    for i, j in product(xrange(len(lpb)-1), xrange(len(lrb)-1)):
        area = (lpb[i+1] - lpb[i]) * (lrb[i+1] - lrb[i])
        k = np.random.poisson(np.exp(lnrate[i, j]) * area)
        if k == 0:
            continue
        entry = np.vstack((np.random.uniform(lpb[i], lpb[i+1], k),
                           np.random.uniform(lrb[j], lrb[j+1], k))).T
        catalog = np.concatenate((catalog, entry), axis=0)

    # Add in some observational uncertainties.
    catalog = np.exp(catalog)
    # err = np.vstack([np.zeros(len(catalog)), 0.1 * catalog[:, 1]]).T
    err = np.vstack([np.zeros(len(catalog)), 0.33 * catalog[:, 1]]).T
    catalog += err * np.random.randn(*(err.shape))

    # Save the catalog.
    pickle.dump((catalog, err), open(os.path.join(bp, "catalog.pkl"), "wb"),
                -1)

    dataset = Dataset.sample(catalog, err, samples=100, censor=censor,
                             functions=[np.log, np.log])
    print("{0} entries in catalog".format(len(catalog)))

    # Plot the actual rate function.
    pl.figure()
    pl.pcolor(lpb, lrb, np.exp(censor.lncompleteness[1:-1, 1:-1].T),
              cmap="gray")
    pl.plot(dataset.catalogs[:, :, 0], dataset.catalogs[:, :, 1], ".r", ms=3,
            alpha=0.3)
    pl.plot(np.log(catalog[:, 0]), np.log(catalog[:, 1]), ".b", ms=5)
    pl.colorbar()
    pl.xlim(min(lpb), max(lpb))
    pl.ylim(min(lrb), max(lrb))
    pl.savefig(os.path.join(bp, "true-rate.png"))
    pl.savefig(os.path.join(bp, "true-rate.pdf"))

    # Build the binned model.
    bins = [x, y]
    print("Run inference on a grid with shape: {0}"
          .format([len(b)-1 for b in bins]))
    pop = Population(bins, censor.bins, lnnorm=truth[0])
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

    figs = pop.plot(v,
                    labels=["$\ln T/\mathrm{days}$", "$\ln R/R_\oplus$"],
                    top_axes=["$T\,[\mathrm{days}]$", "$R\,[R_\oplus]$"],
                    literature=literature)
    figs[0].savefig(os.path.join(bp, "vmax-period.png"))
    figs[0].savefig(os.path.join(bp, "vmax-period.pdf"))
    figs[1].savefig(os.path.join(bp, "vmax-radius.png"))
    figs[1].savefig(os.path.join(bp, "vmax-radius.pdf"))

    pl.clf()
    pl.pcolor(bins[0], bins[1], grid.T, cmap="gray")
    pl.colorbar()
    pl.xlim(min(lpb), max(lpb))
    pl.ylim(min(lrb), max(lrb))
    pl.savefig(os.path.join(bp, "vmax-grid.png"))
    pl.savefig(os.path.join(bp, "vmax-grid.pdf"))

    # Maximize the likelihood.
    def nll(p, huge=1e14):
        ll = model(p)
        if not np.isfinite(ll):
            return huge
        return -ll

    p0 = pop.initial()
    print("Initial ln-prob = {0}".format(model.lnprob(p0)))

    # Set up the sampler.
    ndim, nwalkers = len(p0), 200
    pos = [p0 + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    print("Sampling {0} dimensions with {1} walkers".format(ndim, nwalkers))

    # Make sure that all the initial positions have finite probability.
    finite = np.isfinite(map(model.lnprob, pos))
    assert np.all(finite), "{0}".format(np.sum(finite))

    # Set up the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model)
    sampler.run_mcmc(pos, 10000)

    # Restart.
    if False:
        print("Restarting")
        pickle.dump(sampler,
                    open(os.path.join(bp, "burnin.pkl"), "wb"), -1)
        w, n = np.unravel_index(np.argmax(sampler.lnprobability),
                                sampler.lnprobability.shape)
        p0 = sampler.chain[w, n, :]
        pos = [p0 + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
        finite = np.isfinite(map(model.lnprob, pos))
        assert np.all(finite), "{0}".format(np.sum(finite))
        sampler.reset()
        sampler.run_mcmc(pos, 10000)

    pickle.dump((pop, dataset, censor, sampler),
                open(os.path.join(bp, "results.pkl"), "wb"), -1)

    # Subsample the chain.
    burnin = 8000
    samples = sampler.chain[:, burnin:, :]
    samples = samples.reshape((-1, samples.shape[-1]))
    subsamples = samples[np.random.randint(len(samples), size=100)]

    # Plot the results.
    figs = pop.plot(subsamples,
                    labels=["$\ln T/\mathrm{days}$", "$\ln R/R_\oplus$"],
                    top_axes=["$T\,[\mathrm{days}]$", "$R\,[R_\oplus]$"],
                    literature=literature)
    figs[0].savefig(os.path.join(bp, "period.png"))
    figs[0].savefig(os.path.join(bp, "period.pdf"))
    figs[1].savefig(os.path.join(bp, "radius.png"))
    figs[1].savefig(os.path.join(bp, "radius.pdf"))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        main([sys.argv[1], None], state=pickle.load(open(sys.argv[2])))
    else:
        main([sys.argv[1], None])
