#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["BrokenPowerLaw", "Histogram", "Population", "SeparablePopulation",
           "NormalizedPopulation", "Dataset", "CensoringFunction"]

import numpy as np
from itertools import izip
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

        self.ln_bin_widths = np.log(self.bins[1:] - self.bins[:-1])
        self.nbins = len(self.ln_bin_widths)
        self.inds = np.digitize(0.5*(self.base[1:]+self.base[:-1]),
                                self.bins) - 1
        assert np.all((self.inds >= 0) * (self.inds < len(self.bins)))

    def __len__(self):
        return self.nbins - 1

    def initial(self):
        v = np.zeros(self.nbins)
        v -= logsumexp(v + self.ln_bin_widths)
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

    def __init__(self, log_per_bins, log_rp_bins,
                 log_per_base=None, log_rp_base=None):
        self.log_per_bins = np.atleast_1d(log_per_bins)
        self.log_rp_bins = np.atleast_1d(log_rp_bins)
        if log_per_base is None:
            self.log_per_base = np.array(self.log_per_bins)
        else:
            self.log_per_base = np.atleast_1d(log_per_base)
        if log_rp_base is None:
            self.log_rp_base = np.array(self.log_rp_bins)
        else:
            self.log_rp_base = np.atleast_1d(log_rp_base)

        # Make sure that the resampling is sane.
        assert (self.log_per_bins[0] == self.log_per_base[0])
        assert (self.log_per_bins[-1] == self.log_per_base[-1])
        assert (self.log_rp_bins[0] == self.log_rp_base[0])
        assert (self.log_rp_bins[-1] == self.log_rp_base[-1])

        # Figure out which bins the base grid sits in. This will only be exact
        # when the grids line up perfectly.
        p = self.log_per_base
        rp = self.log_rp_base
        ix = np.digitize(0.5*(p[1:]+p[:-1]), self.log_per_bins) - 1
        assert np.all((ix >= 0) * (ix < len(self.log_per_bins)))
        iy = np.digitize(0.5*(rp[1:]+rp[:-1]), self.log_rp_bins) - 1
        assert np.all((iy >= 0) * (iy < len(self.log_rp_bins)))
        self.iy, self.ix = np.meshgrid(iy, ix)

        # Compute the cell centers and areas.
        p = self.log_per_bins
        rp = self.log_rp_bins
        ir, ip = np.meshgrid(0.5*(rp[1:]+rp[:-1]), 0.5*(p[1:]+p[:-1]))
        self.cell_coords = np.vstack((ip.flatten(), ir.flatten())).T
        self.ln_cell_area = (np.log(p[1:]-p[:-1])[:, None] +
                             np.log(rp[1:]-rp[:-1])[None, :])
        self.grid_shape = self.ln_cell_area.shape
        self.ln_cell_area = self.ln_cell_area.flatten()
        self.ncells = np.prod(len(self.ln_cell_area))

        # Allocate the cache as empty.
        self._cache_key = None
        self._cache_val = None

    def __len__(self):
        return self.ncells + 2

    def initial(self):
        v = np.zeros(self.ncells+3)
        v[3:] -= logsumexp(v[3:] + self.ln_cell_area)
        v[:3] = [0.5, -3.0, -3.0]
        return v[:-1]

    def _get_grid(self, theta):
        k = tuple(theta[3:])
        if k == self._cache_key:
            return self._cache_val

        # Compute the integral over the first N-1 cells.
        norm = logsumexp(theta[3:] + self.ln_cell_area[:-1])
        if norm >= 0.0:
            return None

        # Compute the height of the last cell and cache the heights.
        v = np.log(1.0 - np.exp(norm)) - self.ln_cell_area[-1]
        self._cache_key = k
        self._cache_val = np.append(theta[3:], v).reshape(self.grid_shape)

        return self._cache_val

    def evaluate(self, theta):
        grid = self._get_grid(theta)
        if grid is None:
            return None
        return grid[self.ix, self.iy]

    def lnprior(self, theta):
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

    def __init__(self, log_per_dist, log_rp_dist):
        self.log_per_dist = log_per_dist
        self.log_rp_dist = log_rp_dist
        self.npars = len(self.log_per_dist) + len(self.log_rp_dist)
        super(SeparablePopulation, self).__init__(self.log_per_dist.base,
                                                  self.log_rp_dist.base)

    def __len__(self):
        return self.npars

    def initial(self):
        return np.append(self.log_per_dist.initial(),
                         self.log_rp_dist.initial())

    def evaluate(self, theta):
        assert len(theta) == self.npars
        n = len(self.log_per_dist)

        # Compute the period rate.
        ln_rate_per = self.log_per_dist(theta[:n])
        if ln_rate_per is None:
            return None

        # Compute the radius rate.
        ln_rate_rp = self.log_rp_dist(theta[n:])
        if ln_rate_rp is None:
            return None

        return ln_rate_per[:, None] + ln_rate_rp[None, :]

    def lnprior(self, theta):
        n = len(self.log_per_dist)
        lp = self.log_per_dist.lnprior(theta[:n])
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_rp_dist.lnprior(theta[n:])


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

    def __init__(self, log_per_obs, log_rp_obs, censor):
        self.log_per_obs = np.atleast_1d(log_per_obs)
        self.log_rp_obs = np.atleast_1d(log_rp_obs)
        self.censor = censor

        # Deal with cases where we only have means not samples of radius.
        if len(self.log_rp_obs.shape) == 1:
            self.log_rp_obs = np.atleast_2d(self.log_rp_obs).T

        # Pre-compute the bin indexes for the data-points.
        self.log_per_ind = np.digitize(self.log_per_obs, censor.log_per_bins)
        self.log_rp_ind = (np.digitize(self.log_rp_obs.flatten(),
                                       censor.log_rp_bins)
                           .reshape(self.log_rp_obs.shape))

        # We'll have to remove all the candidates that have no points within
        # the grid or in cells of finite completeness. This is a HACK but I
        # can't think of a better way. The results shouldn't be sensitive to
        # this...
        s = censor.lncompleteness.shape
        q = -np.inf + np.zeros((s[0]+2, s[1]+2))
        q[1:-1, 1:-1] = censor.lnprob_grid
        m = ((self.log_per_ind > 0)[:, None] *
             (self.log_per_ind < len(censor.log_per_bins))[:, None] *
             (self.log_rp_ind > 0) *
             (self.log_rp_ind < len(censor.log_rp_bins)) *
             np.isfinite(q[self.log_per_ind[:, None], self.log_rp_ind]))
        m = np.any(m, axis=1)
        self.log_per_obs = self.log_per_obs[m]
        self.log_rp_obs = self.log_rp_obs[m, :]
        self.log_per_ind = self.log_per_ind[m]
        self.log_rp_ind = self.log_rp_ind[m, :]

        # Pre-compute the "interim prior" weights for each sample.
        w = -np.inf + np.zeros((s[0]+2, s[1]+2))
        w[1:-1, 1:-1] = censor.lnprob_grid
        self.lnw = np.sum(w[self.log_per_ind[:, None], self.log_rp_ind],
                          axis=0)

        # Only include "catalogs" with non-zero prior probability. TOTAL HACK.
        m = np.isfinite(self.lnw)
        self.log_rp_obs = self.log_rp_obs[:, m]
        self.log_rp_ind = self.log_rp_ind[:, m]
        self.lnw = self.lnw[m]


class ProbabilisticModel(object):

    def __init__(self, dataset, population):
        self.dataset = dataset
        self.population = population

    def lnlike(self, theta):
        # Evaluate the population rate.
        lnrate = self.population.evaluate(theta)
        if lnrate is None:
            return -np.inf

        # Add in the censoring function.
        censor = self.dataset.censor
        lnrate += censor.lnprob_grid

        if self.population.poisson:
            norm = np.exp(logsumexp(lnrate + censor.ln_cell_area))
        else:
            lnrate -= logsumexp(lnrate + censor.ln_cell_area)
            norm = 0.0

        # Deal with points outside the range.
        s = lnrate.shape
        q = -np.inf + np.zeros((s[0]+2, s[1]+2))
        q[1:-1, 1:-1] = lnrate

        if self.dataset.log_rp_ind.shape[1] == 1:
            return np.sum(q[self.dataset.log_per_ind,
                            self.dataset.log_rp_ind[:, 0]]) - norm

        ll = q[self.dataset.log_per_ind[:, None], self.dataset.log_rp_ind]
        ll = np.sum(ll, axis=0) - norm - self.dataset.lnw
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


class CensoringFunction(object):

    def __init__(self, log_per_samp, log_rp_samp, is_found_samp,
                 log_per_range=None, log_rp_range=None,
                 nbin_log_per=16, nbin_log_rp=64,
                 transit_lnprob0=np.log(0.0507), ln_period0=np.log(1.0)):
        # Compute the default ranges.
        if log_per_range is None:
            log_per_range = (log_per_samp.min(), log_per_samp.max())
        if log_rp_range is None:
            log_rp_range = (log_rp_samp.min(), log_rp_samp.max())

        # Build the grids.
        self.log_per_bins = np.linspace(log_per_range[0], log_per_range[1],
                                        nbin_log_per+1)
        self.log_rp_bins = np.linspace(log_rp_range[0], log_rp_range[1],
                                       nbin_log_rp+1)

        # Pre-compute the cell areas.
        self.ln_cell_area = (np.log(self.log_per_bins[1:]
                                    - self.log_per_bins[:-1])[:, None] +
                             np.log(self.log_rp_bins[1:]
                                    - self.log_rp_bins[:-1])[None, :])

        # Histogram the injections and the injections that were recovered.
        img_yes, foo, bar = np.histogram2d(log_per_samp[is_found_samp],
                                           log_rp_samp[is_found_samp],
                                           (self.log_per_bins,
                                            self.log_rp_bins))
        img_all, foo, bar = np.histogram2d(log_per_samp,
                                           log_rp_samp,
                                           (self.log_per_bins,
                                            self.log_rp_bins))

        # Compute the completeness fraction dealing with zero injections
        # by asserting that the completeness is zero there.
        self.lncompleteness = -np.inf + np.zeros(img_yes.shape, dtype=float)
        m = img_all > 0
        self.lncompleteness[m] = np.log(img_yes[m]) - np.log(img_all[m])

        # Pre-compute transit probability on the period grid.
        cen = 0.5*(self.log_per_bins[1:] + self.log_per_bins[:-1])
        self.transit_lnprob = transit_lnprob0 - 2.*(cen - ln_period0)/3

        # Build the full ln-probability grid.
        self.lnprob_grid = self.transit_lnprob[:, None] + self.lncompleteness

    def evaluate(self, log_per, log_rp):
        s = self.lnprob_grid.shape
        q = -np.inf + np.zeros((s[0]+2, s[1]+2))
        q[1:-1, 1:-1] = self.lnprob_grid
        return q[np.digitize(log_per, self.log_per_bins),
                 np.digitize(log_rp, self.log_rp_bins)]
