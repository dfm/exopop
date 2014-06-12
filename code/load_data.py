#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module eases interaction with the datasets and monkey patches the import
path. It's "research code"... give me a break!

"""

from __future__ import division, print_function

import os
import sys
import h5py
import numpy as np
import pandas as pd

d = os.path.dirname
base = d(os.path.abspath(__file__))
sys.path.insert(0, base)
from population import SavedCensoringFunction

bp = os.path.join(d(base), "data")

# Hard-coded transit probability scaling.
P = 10.0
transit_lnprob0 = -2.98353340397
ln_period0 = np.log(P)

# Un-comment the following lines to recompute the transit probability if you
# get a list of Petigura's injections.
# G = 2945.4625385377644
# stars = pd.read_hdf(os.path.join(bp, "stlr.h5"), "stlr")
# transit_lnprobs = np.log(stars.Rstar) + (np.log(4*np.pi*np.pi)
#                                          - np.log(G*P*P*stars.Mstar)) / 3
# transit_lnprob0 = np.median(transit_lnprobs)


def load_detection_efficiency():
    """
    Load a pre-computed detection efficiency grid.

    """
    with h5py.File(os.path.join(bp, "completeness.h5"), "r") as f:
        bins = [f["ln_period_bin_edges"][...],
                f["ln_radius_bin_edges"][...]]
        lnprob = f["ln_detect_eff"][...]
        lncompleteness = f["ln_completeness"][...]
    return SavedCensoringFunction(bins, lnprob, lncompleteness)


def load_completenes_sim(rp_func=np.log, per_rng=np.log([5, 400]),
                         rp_rng=None, K=50000):
    """
    This function will only work if you request Petigura's completeness
    simulations from him and save them in the ``data`` directory.

    """
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
