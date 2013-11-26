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

pgm = daft.PGM([3, 2.5], origin=[-1, 1])

pgm.add_plate(daft.Plate([-0.5, 1.4, 2.0, 1.1], label=r"$n=1,\cdots,N$",
                         position="bottom right"))

pgm.add_node(daft.Node("hyperhyper", r"\hyperhyper", -0.5, 3.0, fixed=True))
pgm.add_node(daft.Node("hyper", r"$\hyper$", 0.5, 3.0))
pgm.add_node(daft.Node("local", r"$\local_n$", 0, 2))
pgm.add_node(daft.Node("data", r"$\data_n$", 1, 2, observed=True))

pgm.add_edge("hyperhyper", "hyper")
pgm.add_edge("hyper", "local")
pgm.add_edge("hyper", "data")
pgm.add_edge("local", "data")

pgm.render()
pgm.figure.savefig("abstract.png", dpi=150)
pgm.figure.savefig("abstract.pdf")
