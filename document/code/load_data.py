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

G = 2945.4625385377644
P = 10.0
# stars = pd.read_hdf(os.path.join(bp, "stlr.h5"), "stlr")
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


def load_candidates():
    lines = open(os.path.join(bp, "table_ekoi836.tex")).readlines()
    data = np.array([[l.split("&")[i] for i in (0, 2, 12, 13)] for l in lines
                     if l.split("&")[4].strip() == "P"],
                    dtype=float)
    return (np.array(data[:, 0], dtype=int), data[:, 1:3],
            np.vstack([np.zeros(len(data)), data[:, 3]]).T)


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

    return (ep_p_lnbins, ep_p_pdf), (ep_Rp_lnbins, ep_Rp_pdf)
