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
import numpy as np
import cPickle as pickle
from itertools import product
from scipy.misc import logsumexp

from load_data import transit_lnprob0, ln_period0, load_completenes_sim
from population import (CensoringFunction, SeparablePopulation, Histogram,
                        BrokenPowerLaw)


def main(args, state=None, smooth=False):
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
    p_vals = np.log(np.array([8.9, 13.7, 15.8, 15.2, 15., 14.8]))
    r_vals = np.log(np.array([11, 11.5, 12, 14.2, 18.6, 5.9, 1.9, 1, 0.9, 0.7,
                              0.5, 0.5]))

    # Normalize the underlying distribution.
    p_vals -= logsumexp(p_vals + np.log(np.diff(x)))
    r_vals -= logsumexp(r_vals + np.log(np.diff(y)))

    # Build a synthetic population.
    norm = 10.7
    if smooth:
        truth = [norm, 0.5, -0.2, 4.0, 0.8, -1.5, 1.0]
        pdist = BrokenPowerLaw(lpb)
        rdist = BrokenPowerLaw(lrb)
    else:
        truth = np.concatenate([[norm], p_vals, r_vals])
        pdist = Histogram(x, base=lpb)
        rdist = Histogram(y, base=lrb)
    pop0 = SeparablePopulation([pdist, rdist], lnnorm=truth[0])
    open(os.path.join(bp, "gamma.txt"), "w").write(
        "{0}".format(pop0.get_lnrate(truth, [np.log(365), 0.0])[0]))
    assert 0

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
    err = np.vstack([np.zeros(len(catalog)), 0.25 * catalog[:, 1]]).T
    catalog += err * np.random.randn(*(err.shape))
    print(len(catalog))

    truth = [(pdist.base, np.exp(pdist(truth[1:1+len(pdist)]))),
             (rdist.base, np.exp(rdist(truth[1+len(pdist):])))]

    # Save the catalog.
    pickle.dump((catalog, err, truth),
                open(os.path.join(bp, "catalog.pkl"), "wb"), -1)


if __name__ == "__main__":
    import sys
    args = list(sys.argv)
    try:
        smooth = args.index("--smooth")
    except ValueError:
        smooth = False
    else:
        args.pop(smooth)
        smooth = True

    if len(args) > 2:
        main([args[1], None], state=pickle.load(open(args[2])),
             smooth=smooth)
    else:
        main([args[1], None], smooth=smooth)
