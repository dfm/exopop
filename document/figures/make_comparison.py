#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator

values = [
    ("Petigura \emph{et al.}\ (2013)", 0.119, 0.046, 0.035),
    ("Dong \& Zhu (2013)", 0.086, 0.048, 0.031),
    ("linear extrapolation", 0.072, 0.087, 0.046),
    ("negligible uncertainties",
     0.0397591956934, 0.0309881584307, 0.0190620520124),
    ("Foreman-Mackey \emph{et al.}\ (2014)",
     0.0188817594233, 0.0193141328059, 0.0104440671941),
]

fig = pl.figure(figsize=(8, 4))
for i, v in enumerate(values):
    pl.plot(np.log(v[1]), i, "ok")
    pl.plot(np.log([v[1]+v[2], v[1]-v[3]]), [i, i], "k")

pl.gca().set_yticklabels([""] + [v[0] for v in values])
pl.xlabel(r"$\ln\Gamma_\oplus$")
fig.subplots_adjust(left=0.48, bottom=0.17, right=0.97, top=0.98)
pl.ylim(4.5, -0.5)
pl.gca().xaxis.set_major_locator(MaxNLocator(5))
pl.savefig("comparison.pdf")
