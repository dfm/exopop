#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["BrokenPowerLaw", "Histogram", "Population", "SeparablePopulation",
           "NormalizedPopulation", "Dataset", "CensoringFunction"]

from itertools import izip

import numpy as np
import matplotlib.pyplot as pl
from scipy.misc import logsumexp
from scipy.linalg import cho_factor, cho_solve


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
        self.base = bins
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

        grid = self._get_grid(theta)
        if grid is None:
            return -np.inf

        # Compute the Gaussian process prior.
        y = grid.flatten()
        y -= np.mean(y)
        d = (self.cell_coords[:, None, :] - self.cell_coords[None, :, :])**2
        K = np.exp(theta[0] - 0.5 * np.sum(d / np.exp(theta[1:3]), axis=2))
        K += np.diag(1e-10 * np.ones_like(y))
        factor, flag = cho_factor(K)
        logdet = np.sum(2*np.log(np.diag(factor)))
        return -0.5 * (np.dot(y, cho_solve((factor, flag), y)) + logdet)

    def plot(self, thetas, ep=None, alpha=0.5, rp_label="\ln R"):
        thetas = np.atleast_2d(thetas)

        # Set up figures and axes.
        fig_per = pl.figure(figsize=(6, 5))
        ax_per = fig_per.add_subplot(111)
        fig_rp = pl.figure(figsize=(6, 5))
        ax_rp = fig_rp.add_subplot(111)
        for fig in [fig_per, fig_rp]:
            fig.subplots_adjust(left=0.16, bottom=0.15, right=0.98, top=0.97)

        if ep is None:
            rinds = np.ones(len(self.log_rp_bins), dtype=bool)
        else:
            rinds = ((self.log_rp_bins <= ep[1].max())
                     * (self.log_rp_bins >= ep[1].min()))

        # Loop over samples and plot the projections.
        for theta in thetas:
            grid = self.evaluate(theta)

            z_per = self.log_rp_bins
            y_per = logsumexp(grid + np.log(z_per[1:]-z_per[:-1])[None, :],
                              axis=1)
            z_rp = self.log_per_bins
            y_rp = logsumexp(grid + np.log(z_rp[1:]-z_rp[:-1])[:, None],
                             axis=0)
            y_rp = y_rp[rinds[1:] * rinds[:-1]]

            for ax, x, y in izip([ax_per, ax_rp],
                                 [self.log_per_bins, self.log_rp_bins[rinds]],
                                 [y_per, y_rp]):
                y -= logsumexp(y + np.log(x[1:] - x[:-1]))
                x = np.array(zip(x[:-1], x[1:])).flatten()
                y = np.array(zip(y, y)).flatten()
                ax.plot(x, np.exp(y), "k", alpha=alpha)

        # Plot Erik's values if given.
        if ep is not None:
            ax_rp.plot(0.5*(ep[1][1:]+ep[1][:-1]), ep[3], ".", color="r",
                       ms=8)
            ax_per.plot(0.5*(ep[0][1:]+ep[0][:-1]), ep[2], ".", color="r",
                        ms=8)

        ax_per.set_xlim(np.min(self.log_per_bins),
                        np.max(self.log_per_bins))
        ax_per.set_ylim(np.array((-0.1, 1.0))*(ax_per.get_ylim()[1]))
        ax_per.set_ylabel("$p(\ln P)$")
        ax_per.set_xlabel("$\ln P$")
        ax_per.axhline(0.0, color="k", alpha=0.3)

        ax_rp.set_xlim(np.min(ep[1]), np.max(ep[1]))
        ax_rp.set_ylim(np.array((-0.1, 1.0))*(ax_rp.get_ylim()[1]))
        ax_rp.set_ylabel("$p({0})$".format(rp_label))
        ax_rp.set_xlabel("${0}$".format(rp_label))
        ax_rp.axhline(0.0, color="k", alpha=0.3)

        return fig_per, fig_rp


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


class NormalizedPopulation(object):
    poisson = True

    def __init__(self, ln_norm, base_population):
        self.ln_norm = ln_norm
        self.base_population = base_population

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

    def lnlike(self, theta):
        # Evaluate the population rate.
        lnrate = self.population.evaluate(theta)
        if lnrate is None:
            return -np.inf

        # Compute the slice indexing the non-zero censoring region.
        center = [slice(1, -1)] * len(lnrate.shape)

        # Compute the censoring ln-probability.
        q = self.censor.lnprob
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
