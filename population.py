#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["BrokenPowerLaw", "Histogram", "Population", "SeparablePopulation",
           "NormalizedPopulation", "Dataset", "CensoringFunction"]

from itertools import izip

import numpy as np

from scipy.misc import logsumexp
from scipy.linalg import cho_factor, cho_solve

import matplotlib.pyplot as pl
from matplotlib.ticker import ScalarFormatter


class BrokenPowerLaw(object):

    def __init__(self, bins, alpha_range=(-10, 10), beta_range=(-10, 10),
                 logc_range=None):
        self.bins = np.atleast_1d(bins)
        self.base = np.array(self.bins)
        self.logx = 0.5*(bins[1:] + bins[:-1])
        self.ln_bin_widths = np.log(self.bins[1:] - self.bins[:-1])
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        if logc_range is None:
            logc_range = (np.min(bins)-2, np.max(bins)+2)
        self.logc_range = logc_range

    def __len__(self):
        return 3

    def initial(self):
        return np.array([0.0, 0.0, np.mean(self.bins)])

    def lnprior(self, theta):
        alpha, beta, logc = theta
        if not self.alpha_range[0] < alpha < self.alpha_range[1]:
            return -np.inf
        if not self.beta_range[0] < beta < self.beta_range[1]:
            return -np.inf
        if not self.logc_range[0] < logc < self.logc_range[1]:
            return -np.inf
        return 0.0

    def __call__(self, theta):
        alpha, beta, logc = theta
        x, xc = np.exp(self.logx), np.exp(logc)
        v = alpha*self.logx + (beta-alpha)*np.log(0.5*(xc+x)) - beta*logc
        v -= logsumexp(v + self.ln_bin_widths)
        return v


class Histogram(object):

    def __init__(self, bins, base=None):
        self.bins = np.atleast_1d(bins)
        if base is None:
            self.base = np.array(bins)
        else:
            self.base = np.atleast_1d(base)
        assert self.bins[0] == self.base[0]
        assert self.bins[-1] == self.base[-1]

        self.ln_bin_widths = np.log(np.diff(self.bins))
        assert np.all(np.isfinite(self.ln_bin_widths))
        self.nbins = len(self.ln_bin_widths)
        self.inds = np.digitize(0.5*(self.base[1:]+self.base[:-1]),
                                self.bins) - 1
        assert np.all((self.inds >= 0) * (self.inds < len(self.bins)))

    def __len__(self):
        return self.nbins - 1

    def initial(self):
        v = np.zeros(self.nbins) - logsumexp(self.ln_bin_widths)
        return v[:-1]

    def lnprior(self, theta):
        return 0.0

    def __call__(self, theta):
        norm = logsumexp(theta + self.ln_bin_widths[:-1])
        if norm >= 0.0:
            return None

        # Compute the height of the last cell.
        v = np.log(1.0 - np.exp(norm)) - self.ln_bin_widths[-1]
        ln_heights = np.append(theta, v)

        return ln_heights[self.inds]


class Population(object):
    poisson = False

    def __init__(self, bins, base=None):
        self.bins = bins
        if base is None:
            base = bins
        self.base = base
        self.shape = np.array(map(len, bins)) - 1

        # Make sure that the resampling is sane.
        assert all([b1[0] == b2[0] for b1, b2 in izip(bins, base)])
        assert all([b1[-1] == b2[-1] for b1, b2 in izip(bins, base)])

        # Figure out which bins the base grid sits in. This will only be exact
        # when the grids line up perfectly.
        inds = [np.digitize(bs[:-1]+0.5*np.diff(bs), bn)-1
                for bn, bs in izip(bins, base)]
        assert all([np.all((ix >= 0) * (ix < len(b)))
                    for ix, b in izip(inds, bins)])
        self.inds = np.meshgrid(*inds, indexing="ij")

        # Compute the bin widths, centers and areas using some crazy shit.
        widths = map(np.diff, self.bins)
        self.ln_bin_widths = map(np.log, widths)
        self.bin_centers = [b[:-1] + 0.5*w for b, w in izip(self.bins, widths)]
        self.ln_cell_area = reduce(np.add, np.ix_(*(self.ln_bin_widths)))
        self.ln_cell_area = self.ln_cell_area.flatten()
        self.ncells = len(self.ln_cell_area)

        # Allocate the cache as empty.
        self._cache_key = None
        self._cache_val = None

    def __len__(self):
        return self.ncells - 1

    def initial(self):
        v = np.zeros(self.ncells)
        v -= logsumexp(v + self.ln_cell_area)
        # v[:3] = [0.5, -3.0, -3.0]
        return v[:-1]

    def compute_grid(self, theta):
        # Compute the integral over the first N-1 cells.
        norm = logsumexp(theta + self.ln_cell_area[:-1])
        if norm >= 0.0:
            return None

        # Compute the height of the last cell and cache the heights.
        v = np.log(1.0 - np.exp(norm)) - self.ln_cell_area[-1]
        return np.append(theta, v).reshape(self.shape)

    def _get_grid(self, theta):
        k = tuple(theta)
        if k == self._cache_key:
            return self._cache_val
        self._cache_key = k
        self._cache_val = self.compute_grid(theta)
        return self._cache_val

    def evaluate(self, theta):
        grid = self._get_grid(theta)
        if grid is None:
            return None
        return grid[self.inds]

    def lnprior(self, theta):
        return 0.0

    def plot(self, thetas, labels=None, ranges=None, alpha=0.3,
             literature=None, top_axes=None):
        # Pre-compute the ranges and allowed ranges.
        if ranges is None:
            ranges = [(b.min(), b.max()) for b in self.base]
        bm = []
        vm = []
        for rng, b in izip(ranges, self.base):
            if rng is None:
                rng = (b.min(), b.max())
            m = (b >= rng[0]) * (b <= rng[1])
            bm.append(m)
            vm.append(np.arange(len(m) - 1)[m[:-1] * m[1:]])
        vm = np.meshgrid(*vm, indexing="ij")

        thetas = np.atleast_2d(thetas)
        figs = [pl.figure(figsize=(6, 5)) for b in self.base]
        axes = [f.add_subplot(111) for f in figs]

        lbw0 = map(np.log, map(np.diff,
                               [b[m] for b, m in izip(self.base, bm)]))
        N = len(self.base)
        for theta in thetas:
            grid = self.evaluate(theta)[vm]
            for i, (x, m) in enumerate(izip(self.base, bm)):
                # Compute the volumes along the marginalized axes.
                lbw = list(lbw0)
                lbw.pop(i)
                lnvol = reduce(np.add, np.ix_(*lbw))

                # Build a broadcasting object and axis list to perform the
                # marginalization.
                bc = [slice(None)] * N
                bc[i] = None
                a = range(N)
                a.pop(i)

                # Do the marginalization and normalize.
                y = np.logaddexp.reduce(grid + lnvol[bc], axis=tuple(a))
                y -= logsumexp(y + lbw0[i])

                # Plot the histogram.
                x = np.array(zip(x[m][:-1], x[m][1:])).flatten()
                y = np.array(zip(y, y)).flatten()
                axes[i].plot(x, np.exp(y), "k", alpha=0.3)

        # Plot literature values.
        if literature is not None:
            for ax, (x, y) in izip(axes, literature):
                ax.plot(0.5*(x[1:]+x[:-1]), y, ".r")

        # Set the axis limits.
        for i, (ax, rng) in enumerate(izip(axes, ranges)):
            ax.set_xlim(rng)
            ax.set_ylim(np.array((-0.1, 1.05))*(ax.get_ylim()[1]))
            ax.axhline(0.0, color="k", alpha=0.3)
            if labels is not None:
                ax.set_xlabel(labels[i])
                ax.set_ylabel("$p(${0}$)$".format(labels[i]))

        # Add the linear axes along the top if requested.
        if top_axes is not None:
            for ax, t in izip(axes, top_axes):
                if t is None:
                    continue
                a2 = ax.twiny()
                a2.set_xlim(np.exp(ax.get_xlim()))
                a2.set_xscale("log")
                a2.xaxis.set_major_formatter(ScalarFormatter())
                a2.set_xlabel(t)

        # Format the figures.
        for fig in figs:
            fig.subplots_adjust(left=0.16, bottom=0.15, right=0.95, top=0.85)

        return figs


class SeparablePopulation(Population):
    poisson = False

    def __init__(self, distributions):
        self.distributions = distributions
        self.npars = sum(map(len, distributions))
        super(SeparablePopulation, self).__init__(
            [d.base for d in distributions])

    def __len__(self):
        return self.npars

    def initial(self):
        return np.concatenate([d.initial() for d in self.distributions])

    def compute_grid(self, theta):
        n = 0
        axes = []
        for d in self.distributions:
            axes.append(d(theta[n:n+len(d)]))
            if axes[-1] is None:
                return None
            n += len(d)
        return reduce(np.add, np.ix_(*axes))

    def lnprior(self, theta):
        n = 0
        lp = 0.0
        for d in self.distributions:
            lp += d.lnprior(theta[n:n+len(d)])
            if not np.isfinite(lp):
                return -np.inf
            n += len(d)
        return lp


class SmoothPopulation(object):
    poisson = False

    def __init__(self, pars, base_population, eps=1e-10):
        self.pars = np.atleast_1d(pars)
        self.base_population = base_population
        self.base = base_population.base
        self.eps = eps

        # Pre-compute the distance vectors.
        bins = self.base_population.bins
        coords = np.meshgrid(*[0.5*(b[1:] + b[:-1]) for b in bins],
                             indexing="ij")
        coords = np.vstack([c.flatten() for c in coords]).T
        self.dvec = (coords[:, None, :] - coords[None, :, :])**2
        self.ndim = self.dvec.shape[2] + 1

        assert len(self.pars) == self.ndim

    def __len__(self):
        return len(self.base_population) + self.ndim

    def initial(self):
        return np.append(self.pars, self.base_population.initial())

    def evaluate(self, theta):
        return self.base_population.evaluate(theta[self.ndim:])

    def lnprior(self, theta):
        lp = self.base_population.lnprior(theta[self.ndim:])
        if not np.isfinite(lp):
            return -np.inf

        grid = self.base_population._get_grid(theta[self.ndim:])
        if grid is None:
            return -np.inf

        # Compute the Gaussian process prior.
        y = grid.flatten()
        y -= np.mean(y)
        chi2 = np.sum(self.dvec/np.exp(theta[1:self.ndim]), axis=2)
        K = np.exp(theta[0] - 0.5 * chi2)
        K += np.diag(self.eps * np.ones_like(y))
        factor, flag = cho_factor(K)
        logdet = np.sum(2*np.log(np.diag(factor)))
        return -0.5 * (np.dot(y, cho_solve((factor, flag), y)) + logdet)

    def plot(self, thetas, **kwargs):
        thetas = np.atleast_2d(thetas)
        return self.base_population.plot(thetas[:, self.ndim:], **kwargs)


class NormalizedPopulation(object):
    poisson = True

    def __init__(self, ln_norm, base_population):
        self.ln_norm = ln_norm
        self.base_population = base_population
        self.base = base_population.base

    def __len__(self):
        return len(self.base_population) + 1

    def initial(self):
        return np.append(self.ln_norm, self.base_population.initial())

    def evaluate(self, theta):
        v = self.base_population.evaluate(theta[1:])
        if v is None:
            return None
        return theta[0] + v

    def lnprior(self, theta):
        return self.base_population.lnprior(theta[1:])

    def plot(self, thetas, **kwargs):
        thetas = np.atleast_2d(thetas)
        return self.base_population.plot(thetas[:, 1:], **kwargs)


class Dataset(object):
    """

    :param catalogs: ``(ncatalogs, ncandidates, nparams)``
    :param lnweights: ``(ncatalogs, )``

    """

    def __init__(self, catalogs, lnweights=None):
        self.catalogs = np.atleast_3d(catalogs)
        self.ncatalogs, self.ncandidates, self.nparams = self.catalogs.shape
        if lnweights is None:
            self.lnweights = np.zeros(self.ncatalogs)
        else:
            self.lnweights = np.atleast_1d(lnweights)
        assert len(self.lnweights) == self.ncatalogs


class CensoringFunction(object):

    def __init__(self, samples, recovery, bins=32, range=None,
                 transit_lnprob_function=None):
        # Make sure that the samples have the correct format.
        samples = np.atleast_2d(samples)

        # Compute the recovery and injection histograms.
        img_all, self.bins = np.histogramdd(samples, bins=bins, range=range)
        img_yes, tmp = np.histogramdd(samples[recovery], bins=self.bins)

        # Compute the bin widths, centers and areas using some crazy shit.
        widths = map(np.diff, self.bins)
        self.ln_bin_widths = map(np.log, widths)
        self.bin_centers = [b[:-1] + 0.5*w for b, w in izip(self.bins, widths)]
        self.ln_cell_area = reduce(np.add, np.ix_(*(self.ln_bin_widths)))

        # Compute the completeness asserting zero completeness where there
        # were no injections.
        lncompleteness = -np.inf + np.zeros(img_yes.shape, dtype=float)
        m = img_all > 0
        lncompleteness[m] = np.log(img_yes[m]) - np.log(img_all[m])

        # Compute the transit probability if a function was given.
        if transit_lnprob_function is None:
            lnprob = np.array(lncompleteness)
        else:
            args = np.meshgrid(*(self.bin_centers), indexing="ij")
            transit_lnprob = transit_lnprob_function(*args)
            lnprob = lncompleteness + transit_lnprob

        # Expand the completeness and probability grids to have zeros around
        # the edges.
        self.lncompleteness = -np.inf + np.zeros(np.array(img_yes.shape)+2,
                                                 dtype=float)
        self.lncompleteness[[slice(1, -1)] * len(self.bins)] = lncompleteness

        self.lnprob = -np.inf + np.zeros(np.array(img_yes.shape)+2,
                                         dtype=float)
        self.lnprob[[slice(1, -1)] * len(self.bins)] = lnprob

    def index(self, samples):
        return [np.digitize(x, b)
                for x, b in izip(np.atleast_2d(samples).T, self.bins)]

    def get_lnprob(self, samples):
        i = self.index(samples)
        return self.lnprob[i]

    def get_lncompleteness(self, samples):
        i = self.index(samples)
        return self.lncompleteness[i]


class ProbabilisticModel(object):

    def __init__(self, dataset, population, censor):
        self.dataset = dataset
        self.population = population
        self.censor = censor

        c = dataset.catalogs
        s = c.shape
        self.index = censor.index(c.reshape((-1, s[2])))
        self.index = [i.reshape(s[:2]) for i in self.index]

    def get_lnrate(self, theta):
        # Evaluate the population rate.
        lnrate = self.population.evaluate(theta)
        if lnrate is None:
            return None
        return lnrate + self.censor.lnprob[[slice(1, -1)] * len(lnrate.shape)]

    def lnlike(self, theta):
        # Evaluate the population rate.
        lnrate = self.population.evaluate(theta)
        if lnrate is None:
            return -np.inf

        # Compute the slice indexing the non-zero censoring region.
        center = [slice(1, -1)] * len(lnrate.shape)

        # Compute the censoring ln-probability.
        q = np.array(self.censor.lnprob)
        q[center] += lnrate

        if self.population.poisson:
            norm = np.exp(logsumexp(q[center]+self.censor.ln_cell_area))
        else:
            lnrate -= logsumexp(q[center]+self.censor.ln_cell_area)
            norm = 0.0

        # If there is only one sample, we don't need to do the logsumexp.
        if self.dataset.ncatalogs == 1:
            return np.sum(q[self.index]) - norm - self.dataset.lnweights[0]

        # Compute the approximate marginalized likelihood.
        ll = np.sum(q[self.index], axis=1) - norm - self.dataset.lnweights
        return logsumexp(ll)

    def lnprior(self, theta):
        return self.population.lnprior(theta)

    def lnprob(self, theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.lnlike(theta)
        if not np.isfinite(ll):
            return -np.inf
        return ll + lp

    def __call__(self, theta):
        return self.lnprob(theta)
