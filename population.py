#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["BrokenPowerLaw", "Histogram", "Population", "SeparablePopulation",
           "Dataset", "CensoringFunction"]

from itertools import izip

import numpy as np

from scipy.misc import logsumexp
from scipy.linalg import cho_factor, cho_solve

import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
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
        return self.nbins

    def initial(self):
        v = np.zeros(self.nbins) - logsumexp(self.ln_bin_widths)
        return v

    def lnprior(self, theta):
        return 0.0

    def __call__(self, theta):
        return theta[self.inds]


class Population(object):

    def __init__(self, bins, base=None, lnnorm=0.0):
        self.bins = bins
        if base is None:
            base = bins
        self.base = base
        self.shape = np.array(map(len, bins)) - 1
        self.lnnorm = lnnorm

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
        return self.ncells

    def initial(self):
        v = np.zeros(self.ncells)
        v += self.lnnorm - logsumexp(v + self.ln_cell_area)
        return v

    def compute_grid(self, theta):
        return theta.reshape(self.shape)

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
        return np.sum(theta)

    def plot_2d(self, thetas, ranges=None, censor=None, catalog=None,
                labels=None, top_axes=None, literature=None, lit_style={},
                true=None, true_style={}, alpha=0.3):
        assert len(self.base) == 2

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

        # Compute the grid images.
        thetas = np.atleast_2d(thetas)
        grids = np.array([self.evaluate(t)[vm] for t in thetas])

        fig = pl.figure(figsize=(10, 10))
        ax = pl.axes([0.1, 0.1, 0.6, 0.6])

        ax_top = pl.axes([0.1, 0.7, 0.6, 0.2])
        ax_top.set_xticklabels([])
        ax_right = pl.axes([0.7, 0.1, 0.2, 0.6])
        ax_right.set_yticklabels([])

        # Plot the occurence image.
        img = np.median(grids, axis=0).T
        ax.pcolor(self.base[0], self.base[1], np.exp(img),
                  cmap="gray", alpha=0.8)

        # Plot the occurence histograms.
        ys = [logsumexp(grids
                        + np.log(np.diff(self.base[1][bm[1]]))[None, None, :],
                        axis=2),
              logsumexp(grids
                        + np.log(np.diff(self.base[0][bm[0]]))[None, :, None],
                        axis=1)]
        for i, a in enumerate([ax_top, ax_right]):
            x0 = self.base[i][bm[i]]
            x = np.array(zip(x0[:-1], x0[1:])).flatten()

            for y in ys[i]:
                y -= logsumexp(y + np.log(np.diff(x0))[None, :], axis=1)
                y = np.exp(np.array(zip(y, y)).flatten())

                if i:
                    a.plot(y, x, "k", alpha=alpha)
                    a.set_xlim(np.array((-0.1, 1.05))*(a.get_xlim()[1]))
                    a.axvline(0.0, color="k", alpha=0.3)
                    a.xaxis.set_major_locator(MaxNLocator(4))
                else:
                    a.plot(x, y, "k", alpha=alpha)
                    a.set_ylim(np.array((-0.1, 1.05))*(a.get_ylim()[1]))
                    a.axhline(0.0, color="k", alpha=0.3)
                    a.yaxis.set_major_locator(MaxNLocator(4))

        # Plot the completeness contours.
        if censor is not None:
            x, y = censor.bins
            z = np.exp(censor.lncompleteness[1:-1, 1:-1])
            ax.contour(x[:-1]+0.5*np.diff(x), y[:-1]+0.5*np.diff(y), z.T, 3,
                       colors="k", linewidths=2)

        # Plot the data.
        if catalog is not None:
            ax.plot(catalog[:, 0], catalog[:, 1], ".r")

        # Plot literature values.
        lit_style["marker"] = lit_style.get("marker", ".")
        lit_style["color"] = lit_style.get("color", "r")
        lit_style["ls"] = lit_style.get("ls", "None")
        if literature is not None:
            x, y = literature[0]
            ax_top.plot(0.5*(x[1:]+x[:-1]), y, **lit_style)

            x, y = literature[1]
            ax_right.plot(y, 0.5*(x[1:]+x[:-1]), **lit_style)

        # Plot true values.
        true_style["color"] = true_style.get("color", "r")
        true_style["ls"] = true_style.get("ls", "dashed")
        true_style["lw"] = true_style.get("lw", 2)
        if true is not None:
            x, y = true[0]
            x = np.array(zip(x[:-1], x[1:])).flatten()
            y = np.array(zip(y, y)).flatten()
            ax_top.plot(x, y, **true_style)

            x, y = true[1]
            x = np.array(zip(x[:-1], x[1:])).flatten()
            y = np.array(zip(y, y)).flatten()
            ax_right.plot(y, x, **true_style)

        # Set the axes limits.
        ax.set_xlim(ranges[0])
        ax.set_ylim(ranges[1])
        ax_top.set_xlim(ranges[0])
        ax_right.set_ylim(ranges[1])

        if labels is not None:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax_top.set_ylabel("$p(${0}$)$".format(labels[0]))
            ax_right.set_xlabel("$p(${0}$)$".format(labels[1]))

        # Add the linear axes along the top if requested.
        if top_axes is not None:
            a2 = ax_top.twiny()
            a2.set_xlim(np.exp(ax_top.get_xlim()))
            a2.set_xscale("log")
            a2.xaxis.set_major_formatter(ScalarFormatter())
            a2.set_xlabel(top_axes[0])
            a2.xaxis.set_label_coords(0.5, 1.2)

            a2 = ax_right.twinx()
            a2.set_ylim(np.exp(ax_right.get_ylim()))
            a2.set_yscale("log")
            a2.yaxis.set_major_formatter(ScalarFormatter())
            a2.set_ylabel(top_axes[1], rotation=-90)
            a2.yaxis.set_label_coords(1.3, 0.5)

        for a in [ax, ax_top, ax_right]:
            a.xaxis.set_label_coords(0.5, -0.07)
            a.yaxis.set_label_coords(-0.08, 0.5)

        return fig

    def plot(self, thetas, labels=None, ranges=None, alpha=0.3,
             literature=None, lit_style={}, top_axes=None):
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
        lit_style["marker"] = lit_style.get("marker", ".")
        lit_style["color"] = lit_style.get("color", "r")
        lit_style["ls"] = lit_style.get("ls", "None")
        if literature is not None:
            for ax, (x, y) in izip(axes, literature):
                ax.plot(0.5*(x[1:]+x[:-1]), y, **lit_style)

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

    def __init__(self, distributions, lnnorm=0.0):
        self.distributions = distributions
        self.npars = sum(map(len, distributions)) + 1
        super(SeparablePopulation, self).__init__(
            [d.base for d in distributions], lnnorm=lnnorm)

    def __len__(self):
        return self.npars

    def initial(self):
        return np.concatenate([self.lnnorm]
                              + [d.initial() for d in self.distributions])

    def compute_grid(self, theta):
        n = 1
        axes = []
        for d in self.distributions:
            axes.append(d(theta[n:n+len(d)]))
            if axes[-1] is None:
                return None
            n += len(d)
        return theta[0] + reduce(np.add, np.ix_(*axes))

    def lnprior(self, theta):
        n = 1
        lp = 0.0
        for d in self.distributions:
            lp += d.lnprior(theta[n:n+len(d)])
            if not np.isfinite(lp):
                return -np.inf
            n += len(d)
        return lp


class SmoothPopulation(object):

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
        self.ndim = self.dvec.shape[2] + 2

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
        y = grid.flatten() - theta[0]
        chi2 = np.sum(self.dvec/np.exp(theta[2:self.ndim]), axis=2)
        K = np.exp(theta[1] - 0.5 * chi2)
        K += np.diag(self.eps * np.ones_like(y))
        factor, flag = cho_factor(K)
        logdet = np.sum(2*np.log(np.diag(factor)))
        return -0.5 * (np.dot(y, cho_solve((factor, flag), y)) + logdet)

    def plot_2d(self, thetas, *args, **kwargs):
        thetas = np.atleast_2d(thetas)
        return self.base_population.plot_2d(thetas[:, self.ndim:], *args,
                                            **kwargs)

    def plot(self, thetas, **kwargs):
        thetas = np.atleast_2d(thetas)
        return self.base_population.plot(thetas[:, self.ndim:], **kwargs)


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

    @classmethod
    def sample(cls, values, uncertainties, samples=64, functions=None,
               censor=None, tot=10000):
        values = np.atleast_2d(values)
        uncertainties = np.atleast_2d(uncertainties)
        K, N = values.shape
        good = np.ones(K, dtype=bool)
        catalogs = np.empty((samples, K, N))
        lnweights = np.zeros(samples)
        for k, (m, s) in enumerate(izip(values, uncertainties)):
            # Directly sample from the Gaussians.
            if censor is None:
                v = m[None, :] + s[None, :] * np.random.randn(samples, N)
                if functions is not None:
                    for n, f in enumerate(functions):
                        if f is None:
                            continue
                        v[:, n] = f(v[:, n])
                catalogs[:, k, :] = v
                continue

            # Use rejection sampling to sample from the joint posterior.
            v = m[None, :] + s[None, :] * np.random.randn(tot, N)
            if functions is not None:
                for n, f in enumerate(functions):
                    if f is None:
                        continue
                    v[:, n] = f(v[:, n])

            # Compute the acceptance mask.
            lnp = censor.get_lncompleteness(v)
            mask = np.random.rand(len(lnp)) < np.exp(lnp)

            # Make sure that we're getting enough samples for each object.
            if np.sum(mask) < samples:
                print("Dropping candidate at {0}".format(m))
                good[k] = 0
                continue

            inds = np.arange(len(lnp))[mask]
            catalogs[:, k, :] = v[inds[:samples]]
            lnweights += lnp[inds[:samples]]
        return cls(catalogs[:, good], lnweights)


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

        norm = np.exp(logsumexp(q[center]+self.censor.ln_cell_area))

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
