#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
rc("text.latex",
   preamble=r"\input{{{0}}}"
   .format(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vars")))

import daft

pgm = daft.PGM([4.5, 3], origin=[-0.5, 1])

pgm.add_node(daft.Node("population", r"\population", 0.0, 3.0, fixed=True))

pgm.add_node(daft.Node("planet", r"\planet", 1.0, 3.0))
pgm.add_node(daft.Node("planetobs", r"\planetobs", 2.0, 3.0, observed=True))

pgm.add_node(daft.Node("stellar", r"\stellar", 3.0, 3.0))
pgm.add_node(daft.Node("stellarobs", r"$\stellarobs$", 3.0, 2.0,
                       observed=True))

pgm.add_node(daft.Node("isobs", r"\isobs", 1.5, 2.0, observed=True))

pgm.add_node(daft.Node("selection", r"\selection", 0.0, 2.0, fixed=True))

pgm.add_edge("population", "planet")
pgm.add_edge("planet", "planetobs")
pgm.add_edge("planet", "isobs")
pgm.add_edge("stellar", "planetobs")
pgm.add_edge("stellar", "isobs")
pgm.add_edge("stellar", "stellarobs")
pgm.add_edge("selection", "isobs")

pgm.add_plate(daft.Plate([0.4, 1.4, 3.1, 2.2], label=r"stars",
                         position="bottom right"))
pgm.add_plate(daft.Plate([0.5, 1.5, 2, 2], label=r"planets"))

pgm.render()
pgm.figure.savefig("gm.png", dpi=150)
pgm.figure.savefig("gm.pdf")
