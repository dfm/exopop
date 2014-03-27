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


def load_completenes_sim(rp_func=np.log):
    sim = pd.read_hdf(os.path.join(bp, "mcDV.h5"), "mcDV")
    m = sim.found * sim.bDV
    return np.log(sim.inj_P), rp_func(sim.inj_Rp), m


def load_candidates(censor, samples=0, N=10000, rp_func=np.log):
    log_per_obs = []
    log_rp_obs = []
    for line in open(os.path.join(bp, "table_ekoi836.tex")):
        cols = line.split("&")
        log_per = np.log(float(cols[2]))
        log_per_obs.append(log_per)
        if samples > 0:
            m, s = map(float, cols[12:14])
            r = m + s*np.sort(np.random.randn(N))
            lnw = censor.evaluate([log_per], rp_func(r)) - 0.5 * ((r-m)/s)**2
            cummulative = np.cumsum(np.exp(lnw))
            cummulative /= cummulative[-1]
            rnd = np.random.rand(samples)
            r = np.interp(rnd, cummulative, r)
            log_rp_obs.append(rp_func(r))
        else:
            log_rp_obs.append(rp_func(float(cols[12])))

    log_per_obs = np.array(log_per_obs, dtype=float)
    log_rp_obs = np.array(log_rp_obs, dtype=float)
    return population.Dataset(log_per_obs, log_rp_obs, censor)


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
