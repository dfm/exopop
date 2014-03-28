#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import numpy as np
import pandas as pd

d = os.path.dirname
base = d(d(d(os.path.abspath(__file__))))
sys.path.insert(0, base)
import population
population = population

bp = os.path.join(base, "data")

stars = pd.read_hdf(os.path.join(bp, "stlr.h5"), "stlr")
G = 2945.4625385377644
P = 10.0
# transit_lnprobs = np.log(stars.Rstar) + (np.log(4*np.pi*np.pi)
#                                          - np.log(G*P*P*stars.Mstar)) / 3
# transit_lnprob0 = np.median(transit_lnprobs)
transit_lnprob0 = -2.98353340397
ln_period0 = np.log(P)


def load_completenes_sim(rp_func=np.log, per_rng=np.log([5, 400]),
                         rp_rng=None, K=50000):
    if rp_rng is None:
        rp_rng = rp_func([0.5, 16])
    sim = pd.read_hdf(os.path.join(bp, "mcDV.h5"), "mcDV")
    m = sim.found * sim.bDV
    x, y, z = np.log(sim.inj_P), rp_func(sim.inj_Rp), m
    if rp_rng[1] > rp_func(16):
        x = np.append(x, np.random.uniform(per_rng[0], per_rng[1], K))
        y = np.append(y, np.random.uniform(rp_func(16.), rp_rng[1], K))
        z = np.append(z, np.ones(K, dtype=bool))
    return x, y, z


def load_candidates(censor=None, samples=0, N=10000, rp_func=np.log):
    lines = open(os.path.join(bp, "table_ekoi836.tex")).readlines()
    K = len(lines)
    good = np.ones(K, dtype=bool)
    catalogs = np.empty((max(samples, 1), K, 2))
    lnweights = np.zeros(max(samples, 1))
    for k, line in enumerate(lines):
        cols = line.split("&")
        log_per = np.log(float(cols[2]))

        if samples > 0:
            m, s = map(float, cols[12:14])
            catalogs[:, k, 0] = log_per
            if censor is None:
                catalogs[:, k, 1] = rp_func(m+s*np.random.randn(samples))
                continue
            r = m + s*np.random.randn(N)
            r = rp_func(r[r > 0.0])
            v = np.vstack([log_per+np.zeros_like(r), r]).T
            lnp = censor.get_lncompleteness(v)
            mask = np.random.rand(len(lnp)) < np.exp(lnp)
            if np.sum(mask) < samples:
                print("Dropping candidate at R={0}".format(m))
                good[k] = 0
                continue
            inds = np.arange(len(lnp))[mask]
            catalogs[:, k, 1] = r[inds[:samples]]
            lnweights += lnp[inds[:samples]]
        else:
            catalogs[0, k, 0] = log_per
            catalogs[0, k, 1] = rp_func(float(cols[12]))

    return population.Dataset(catalogs[:, good], lnweights)


def load_petigura_bins(mylog=np.log):
    ep_Rp_logbins = 0.5 * np.log10(2) * np.arange(9)
    ep_Rp_lnbins = mylog(10 ** ep_Rp_logbins)
    ep_Rp_values = np.array([12., 14.2, 18.6, 5.9, 1.9, 1.0, 0.9, 0.7])
    norm = np.sum(ep_Rp_values * (ep_Rp_lnbins[1:] - ep_Rp_lnbins[:-1]))
    ep_Rp_pdf = ep_Rp_values / norm

    ep_p_logbins = np.log10([6.25, 12.5, 25, 50, 100])
    ep_p_lnbins = mylog(10 ** ep_p_logbins)
    ep_p_values = np.array([8.9, 13.7, 15.8, 15.2])
    norm = np.sum(ep_p_values * (ep_p_lnbins[1:] - ep_p_lnbins[:-1]))
    ep_p_pdf = ep_p_values / norm

    return ep_p_lnbins, ep_Rp_lnbins, ep_p_pdf, ep_Rp_pdf
