#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import numpy as np
np.random.seed(1000004)

# Newton's constant in $R_\odot^3 M_\odot^{-1} {days}^{-2}$.
G = 2945.4625385377644


def generate_catalog(N, rate=5):
    Rs = 0.5 + 0.7 * np.random.rand(N)
    Ms = 0.5 + 0.7 * np.random.rand(N)
    incl = np.arccos(1-2*np.random.rand(N))

    K = np.random.poisson(rate, N)
    r = [np.random.rand(k) for k in K]
    a = [np.exp(8*np.random.rand(k)) for k in K]
    P = [2*np.pi*np.sqrt(a0**3/(G*M)) for a0, M in zip(a, Ms)]
    di = [np.radians(30*np.random.randn(k)) for k in K]
    b = [a0*np.sin(i+di0) for i, di0, a0 in zip(incl, di, a)]


if __name__ == "__main__":
    generate_catalog(1000)
