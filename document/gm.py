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

pgm = daft.PGM([6, 2.6], origin=[-0.5, 1])

pgm.add_plate(daft.Plate([0.4, 1.1, 4.1, 2.4], label=r"stars $n=1,\ldots,N$",
                         position="bottom right"))
pgm.add_plate(daft.Plate([0.5, 1.4, 2.5, 2], label=r"planets $k=1,\ldots,K$"))

pgm.add_node(daft.Node("rpop", r"\rpop", 0.0, 3.0, fixed=True))
pgm.add_node(daft.Node("selection", r"\selection", 0.0, 2.0, fixed=True))
pgm.add_node(daft.Node("rp", r"$\rp_{kn}$", 1.0, 3.0))
pgm.add_node(daft.Node("ror", r"$\ror_{kn}$", 2.0, 3.0))
pgm.add_node(daft.Node("rorobs", r"$\rorobs_{kn}$", 2.5, 2.5, observed=True))
pgm.add_node(daft.Node("isobs", r"$\isobs_{kn}$", 2, 2, observed=True))

pgm.add_node(daft.Node("Rpop", r"\Rpop", 5, 3.0, fixed=True))
pgm.add_node(daft.Node("Rs", r"$\Rs_{n}$", 3.5, 3.0))
pgm.add_node(daft.Node("Rsobs", r"$\Rsobs_{n}$", 4, 2.5, observed=True))
pgm.add_node(daft.Node("sigmasobs", r"$\sigmasobs_{n}$", 3.5, 2.0,
                       observed=True))

pgm.add_edge("rpop", "rp")
pgm.add_edge("Rpop", "Rs")

pgm.add_edge("Rs", "ror")
pgm.add_edge("rp", "ror")
pgm.add_edge("ror", "rorobs")

pgm.add_edge("Rs", "Rsobs")

pgm.add_edge("ror", "isobs")
pgm.add_edge("sigmasobs", "isobs")
pgm.add_edge("selection", "isobs")

pgm.render()
pgm.figure.savefig("gm.png", dpi=150)
pgm.figure.savefig("gm.pdf")
