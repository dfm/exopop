#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import numpy as np
import cPickle as pickle

from load_data import load_petigura_bins

mylog = np.log
rp_label = "\ln R"
if "--linear" in sys.argv:
    mylog = lambda x: x
    rp_label = "R"

ep = load_petigura_bins(mylog=mylog)
censor, dataset, pop, sampler = pickle.load(open(sys.argv[1]))

samples = sampler.chain
samples = samples.reshape((-1, samples.shape[-1]))

fig_per, fig_rp = pop.plot(samples[np.random.randint(len(samples), size=24)],
                           rp_label=rp_label, ep=ep)
fig_per.savefig(os.path.splitext(sys.argv[1])[0]+"-period.png")
fig_rp.savefig(os.path.splitext(sys.argv[1])[0]+"-radius.png")
