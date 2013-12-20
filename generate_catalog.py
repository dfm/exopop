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
    sigs = 5e-4 + 1e-4 * np.random.rand(N)
    incl = np.arccos(1-2*np.random.rand(N))

    K = np.random.poisson(rate, N)
    r = [np.random.rand(k) for k in K]
    z = [r0/R for r0, R in zip(r, Rs)]
    a = [np.exp(8*np.random.rand(k)) for k in K]
    P = [2*np.pi*np.sqrt(a0**3/(G*M)) for a0, M in zip(a, Ms)]
    di = [np.radians(30*np.random.randn(k)) for k in K]
    b = [a0*np.sin(i+di0)/R for i, di0, a0, R in zip(incl, di, a, Rs)]
    signal = [z0**2*np.sqrt(4.2/P0)*(b0+z0 < 1.0)
              for z0, P0, b0 in zip(z, P, b)]
    print([sum(s/sig>7) for s, sig in zip(signal, sigs)])


if __name__ == "__main__":
    generate_catalog(1000)
