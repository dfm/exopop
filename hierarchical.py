#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import numpy as np

_G = 2945.4625385377644


class Uniform(object):

    def __init__(self, mn, mx):
        assert mx > mn
        self.mn = mn
        self.mx = mx

    def __call__(self, value):
        if np.isscalar(value):
            if self.mn <= value <= self.mx:
                return -np.log(self.mx-self.mn)
            return -np.inf
        m = (self.mn > value) + (value > self.mx)
        r = -np.log(self.mx-self.mn) + np.zeros_like(value)
        r[m] = -np.inf
        return r

    def sample(self, size=1):
        return self.mn + (self.mx-self.mn)*np.random.rand(size)


class Normal(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, std):
        var = std*std
        self._std = std
        self._mh_inv_var = -0.5/var
        self._mh_ln_2pi_var = -0.5 * np.log(2*np.pi*var)

    def __call__(self, value):
        return (value-self.mean)**2*self._mh_inv_var+self._mh_ln_2pi_var

    def sample(self, size=1):
        return self.mean + self._std * np.random.randn(size)


class Power(object):

    def __init__(self, mn, mx, slope):
        self.mn = mn
        self.mx = mx
        self.slope = slope

    def __call__(self, value):
        np1 = self.slope+1
        x0, x1 = self.mn**np1, self.mx**np1
        v = np.log(np1) - np.log(x1-x0) + self.slope * np.log(value)
        if np.isscalar(value):
            if self.mn <= value <= self.mx:
                return v
            return -np.inf
        m = (self.mn > value) + (value > self.mx)
        v[m] = -np.inf
        return v

    def sample(self, size=1):
        y = np.random.rand(size)
        np1 = self.slope+1
        x0, x1 = self.mn**np1, self.mx**np1
        return ((x1 - x0) * y + x0) ** (1.0/np1)


class Selection(object):

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def __call__(self, s2n, b):
        return (0.0 < b < 1.0) * (s2n >= self.cutoff)


class Population(object):

    def __init__(self, mutual_incl_dist, log_period_dist, radius_dist,
                 selection):
        self.mutual_incl_dist = mutual_incl_dist
        self.log_period_dist = log_period_dist
        self.radius_dist = radius_dist
        self.selection = selection

    def generate_systems(self, size, nplanets=20):
        self.systems = [System.sythesize(
            self, 0.5 + np.random.rand(), 0.5 + np.random.rand(),
            np.degrees(np.arccos(np.random.rand()))) for i in range(size)]
        [s.generate_planets(nplanets) for s in self.systems]

    def get_catalog(self):
        return np.array([(i, ) + row for i, system in enumerate(self.systems)
                         for row in system.get_catalog()])


class System(object):

    def __init__(self, population, mass_obs, mass_err, radius_obs, radius_err,
                 sigma, planets=[]):
        self.population = population
        self.mass_obs = mass_obs
        self.mass_err = mass_err
        self.radius_obs = radius_obs
        self.radius_err = radius_err
        self.sigma = sigma
        self.planets = list(planets)

    @classmethod
    def sythesize(cls, population, mass, radius, incl):
        # Draw the observations.
        mass_err = 0.1+0.1*np.random.rand()
        mass_obs = np.abs(mass+mass_err*np.random.randn())
        radius_err = 0.1+0.1*np.random.rand()
        radius_obs = np.abs(radius+radius_err*np.random.randn())
        sigma = 1e-4*np.random.rand()
        self = cls(population, mass_obs, mass_err, radius_obs, radius_err,
                   sigma)
        self.mass = mass
        self.radius = radius
        self.incl = incl
        return self

    def generate_planets(self, nplanets):
        # Generate some planets.
        incls = self.population.mutual_incl_dist.sample(nplanets)
        periods = np.exp(self.population.log_period_dist.sample(nplanets))
        radii = self.population.radius_dist.sample(nplanets)
        self.planets += [Planet.synthesize(self, *args)
                         for args in zip(periods, radii, incls)]

    def get_catalog(self):
        return [(p.period_obs, p.period_err, p.ror_obs, p.ror_err, p.b_obs,
                 p.b_err) for p in self.planets if p.observed]


class Planet(object):

    def __init__(self, system, observed, period_obs=None, period_err=None,
                 ror_obs=None, ror_err=None, b_obs=None, b_err=None):
        self.system = system
        self.observed = observed
        self.period_obs = period_obs
        self.period_err = period_err
        self.ror_obs = ror_obs
        self.ror_err = ror_err
        self.b_obs = b_obs
        self.b_err = b_err

    @classmethod
    def synthesize(cls, system, period, radius, delta_incl, delta_time=4.0):
        incl = delta_incl + system.incl
        a = (_G*period*period*system.mass/(4*np.pi*np.pi)) ** (1./3)
        b = a * np.tan(np.radians(incl)) / system.radius
        ror = radius / system.radius
        signal = np.sqrt(delta_time / period) * ror*ror
        s2n = signal / system.sigma
        observed = system.population.selection(s2n, b)

        if not observed:
            return cls(system, False)

        b_err = 0.1 * np.random.rand()
        b_obs = np.abs(b + b_err*np.random.randn())
        period_err = 1e-5 * np.random.rand()
        period_obs = period + period_err * np.random.randn()
        ror_err = 5e-4 * np.random.rand()
        ror_obs = np.abs(ror + ror_err * np.random.randn())
        return cls(system, True, period_obs, period_err, ror_obs, ror_err,
                   b_obs, b_err)


if __name__ == "__main__":
    population = Population(
        Normal(0.0, 10.0),
        Normal(np.log(50.0), 2.0),
        Power(0.001, 0.5, -1.8),
        Selection(7.0)
    )
    population.generate_systems(10000)
    catalog = population.get_catalog()

    import matplotlib.pyplot as pl
    pl.plot(np.log10(catalog[:, 1]), np.log10(catalog[:, 3]), ".k")
    pl.savefig("dist.png")
