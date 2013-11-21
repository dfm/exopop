#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import numpy as np


class LogNormal(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, size=1):
        return np.exp(self.mean + self.std * np.random.randn(size))

    def __call__(self, x):
        return np.exp(-0.5 * (np.log(x)-self.mean)**2/self.std**2) \
            / np.sqrt(2*np.pi) / self.std


def generate_catalog(N, K, delta, rdist):
    # Generate stellar radii.
    R = 5 * np.random.rand(N)
    R_err = R * (0.1 + 0.1 * np.random.rand(N))
    R_obs = R + R_err * np.random.randn(N)

    # Generate planetary radii.
    r = rdist.sample(N*K).reshape((N, K))
    ror = r / R[:, None]
    ror_err = 1e-4 * np.random.rand(*ror.shape)
    ror_obs = ror + ror_err * np.random.randn(*ror.shape)
    star_id, tmp = np.meshgrid(np.arange(K), np.arange(N))

    # Apply selection.
    q = ror**2 > delta

    return (
        np.array(zip(R_obs, R_err)),
        np.array(zip(star_id[q], ror_obs[q], ror_err[q]))
    )


if __name__ == "__main__":
    import matplotlib.pyplot as pl

    rdist = LogNormal(-5, 1)
    stars, planets = generate_catalog(5000, 20, 1e-3, rdist)

    # Plot observed histogram.
    rstar = stars[:, 0][np.array(planets[:, 0], dtype=int)]
    ror = planets[:, 1]
    r = ror * rstar
    pl.hist(np.log(r), 50, normed=True, histtype="step", color="k")

    # Plot true distribution.
    logr = np.linspace(-8, 1, 5000)
    pl.plot(logr, rdist(np.exp(logr)), "k", lw=2)

    pl.savefig("dist.png")
