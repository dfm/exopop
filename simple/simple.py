#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import numpy as np


class Normal(object):

    def __init__(self, mean, std):
        self.std = std

    def sample(self, size=1):
        return np.abs(self.std * np.random.randn(size))

    def __call__(self, x):
        return 2*np.exp(-0.5*x*x/self.std**2) / np.sqrt(2*np.pi) / self.std


class Histogram(object):

    def __init__(self, bins, values=None):
        nbins = len(bins) - 1
        self.bins = np.array(bins)
        if values is not None:
            assert len(values) == nbins
            self.values = np.array(values)
        else:
            self.values = np.ones(nbins)
        self.normalize()

    def normalize(self):
        norm = np.sum(self.values * (self.bins[1:] - self.bins[:-1]))
        self.values /= norm

    def sample(self, size=1):
        rands = np.random.rand(size)
        cs = np.cumsum(self.values * (self.bins[1:] - self.bins[:-1]))
        return np.interp(rands, np.append(0, cs), self.bins)

    def __call__(self, x):
        n = len(self.bins)
        loc = np.interp(x, self.bins, np.arange(n), left=n-1, right=n-1)
        return np.append(self.values, 0.0)[np.array(np.floor(loc), dtype=int)]


def generate_catalog(N, K, delta, rdist):
    # Generate stellar radii.
    R = 1.0 + 2 * np.random.rand(N)
    R_err = R * (0.1 + 0.1 * np.random.rand(N))
    R_obs = R + R_err * np.random.randn(N)

    # Generate planetary radii.
    r = rdist.sample(N*K).reshape((N, K))
    ror = r / R[:, None]
    ror_err = 1e-4 * np.random.rand(*ror.shape)
    ror_obs = np.abs(ror + ror_err * np.random.randn(*ror.shape))
    star_id, tmp = np.meshgrid(np.arange(K), np.arange(N))

    # Apply selection.
    q = ror**2 > delta

    return (
        np.array(zip(R_obs, R_err)),
        np.array(zip(star_id[q], ror_obs[q], ror_err[q]))
    )


def compute_marginalized_likelihood(K, stars, planets, delta, rdist,
                                    nsamples=500):
    N = len(stars)

    # The likelihood of the stellar data.
    R_obs, R_err = stars[:, 0], stars[:, 1]
    R = R_obs + R_err * np.random.randn(nsamples, N)

    # Sample the radii.
    rmax = np.sqrt(R*R*delta)
    r_null = rmax*np.random.rand(nsamples, N)
    w_null = rdist(r_null)*rmax

    # Sample radii for observed planets.
    counts = np.zeros(N, dtype=int)
    inds = np.array(planets[:, 0], dtype=int)
    counts[inds] += 1
    ror = planets[:, 1]+planets[:, 2]*np.random.randn(nsamples, len(planets))
    r = ror*R[:, inds]
    w = rdist(r)
    w[ror*ror < delta] = 0.0

    weights = (K-counts) * np.log(w_null)
    weights[:, inds] += np.log(w)
    weights = np.sum(weights, axis=1)
    weights -= np.max(weights[np.isfinite(weights)])
    print(weights)
    assert 0

    # Append the unobserved weights.
    r = np.append(r.flatten(), r_null.flatten())
    w = np.append(w.flatten(), (w_null ** (K-counts)).flatten())

    return r, w


if __name__ == "__main__":
    import matplotlib.pyplot as pl

    K = 20
    delta = 1e-1
    true_rdist = Normal(0, 1)
    stars, planets = generate_catalog(5000, K, delta, true_rdist)

    # Plot observed histogram.
    rstar = stars[:, 0][np.array(planets[:, 0], dtype=int)]
    ror = planets[:, 1]
    r = ror * rstar
    pl.hist(r, 100, normed=True, histtype="step", color="k")

    # Plot true distribution.
    x = np.linspace(0, 5, 5000)
    pl.plot(x, true_rdist(x), "k", lw=2)

    # Estimate the r distribution.
    bins = np.linspace(0, 5, 50)
    rdist = Histogram(bins)

    for i in range(5):
        print(i)
        rs, ws = compute_marginalized_likelihood(K, stars, planets, delta,
                                                 rdist)
        rdist.values, tmp = np.histogram(rs, rdist.bins, weights=ws,
                                         normed=True)

        pl.plot(x, rdist(x))

    pl.savefig("dist.png")
