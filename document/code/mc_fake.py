#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from fake_noise import main
from multiprocessing import Pool

fns = map("fake_noise/{0:03d}".format, range(16))
seeds = np.random.randint(50000, size=len(fns))
pool = Pool()
pool.map(main, zip(fns, seeds))
