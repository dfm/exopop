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
from population import (CensoringFunction, BrokenPowerLaw,
                        SeparablePopulation, NormalizedPopulation,
                        ProbabilisticModel, Dataset, Population,
                        BinToBinPopulation)


def main(args, state=None):
    bp, seed = args
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
    per_rng = np.log([5.0, 400.0])
    rp_rng = np.log([0.5, 64.0])
    truth = [11.0, 0.5, -0.2, 4.0, 0.8, -1.5, 1.0]

    # Load the data.
    ln_P_inj, ln_R_inj, recovered = load_completenes_sim(per_rng=per_rng,
                                                         rp_rng=rp_rng)
    tlp = lambda lnp, lnr: transit_lnprob0 - 2.*(lnp - ln_period0)/3

    # Set up the censoring function.
    censor = CensoringFunction(np.vstack((ln_P_inj, ln_R_inj)).T, recovered,
                               bins=(32, 36),
                               range=[per_rng, rp_rng],
                               transit_lnprob_function=tlp)

    # Build a synthetic population.
    lpb, lrb = censor.bins
    pdist = BrokenPowerLaw(lpb)
    rdist = BrokenPowerLaw(lrb)
    pop0 = SeparablePopulation([pdist, rdist])
    pop0 = NormalizedPopulation(truth[0], pop0)

    # Plot the true distributions.
    literature = [(pdist.bins, np.exp(pdist(truth[1:4]))),
                  (rdist.bins, np.exp(rdist(truth[4:])))]
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
    bins = [lpb[::8], lrb[::4]]
    print("Run inference on a grid with shape: {0}".format(map(len, bins)))
    pop = Population(bins, censor.bins)
    pop = NormalizedPopulation(truth[0], pop)
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
    a = map(np.diff, bins)
    lg = np.log(grid)
    areas = np.log(a[0][:, None] * a[1][None, :])
    norm = logsumexp(lg + areas)
    print("norm = {0}".format(norm))
    v = np.append(norm, lg.flatten()[:-1] - norm)
    if pop.evaluate(v) is None:
        print("********** failed: {0}"
              .format(logsumexp(v[1:] + areas.flatten()[:-1])))
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
    results = op.minimize(nll, p0, method="L-BFGS-B", jac=False)
    print(results)
    p0 = results.x
    print("Final ln-prob = {0}".format(model.lnprob(p0)))
    figs = pop.plot(p0,
                    labels=["$\ln T/\mathrm{days}$", "$\ln R/R_\oplus$"],
                    top_axes=["$T\,[\mathrm{days}]$", "$R\,[R_\oplus]$"],
                    literature=literature)
    figs[0].savefig(os.path.join(bp, "ml-period.png"))
    figs[0].savefig(os.path.join(bp, "ml-period.pdf"))
    figs[1].savefig(os.path.join(bp, "ml-radius.png"))
    figs[1].savefig(os.path.join(bp, "ml-radius.pdf"))

    # Update the model with a smoothness prior.
    p0 = np.append([-4, -4], p0)
    pop = BinToBinPopulation([-4, -4], pop)
    model = ProbabilisticModel(dataset, pop, censor)

    # Set up the sampler.
    ndim, nwalkers = len(p0), 100
    pos = [p0 + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    print("Sampling {0} dimensions with {1} walkers".format(ndim, nwalkers))

    # Make sure that all the initial positions have finite probability.
    finite = np.isfinite(map(model.lnprob, pos))
    assert np.all(finite), "{0}".format(np.sum(finite))

    # Set up the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model)
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
        main([sys.argv[1], 1234], state=pickle.load(open(sys.argv[2])))
    else:
        main([sys.argv[1], 1234])
