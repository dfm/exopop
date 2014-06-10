#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import glob
import shutil
from datetime import date
from itertools import imap
from subprocess import check_call

datestamp = date.today().strftime("%m%d")
TMPDIR = "fore" + datestamp
outfn = TMPDIR+".tar"
tex = open("ms.tex", "r").read()

try:
    os.makedirs(TMPDIR)
except os.error:
    pass


def rename(fn):
    a, b = os.path.split(fn)
    return os.path.split(a)[1] + "-" + b, fn


for a, b in imap(rename, glob.glob("figures/*/*.pdf")):
    shutil.copyfile(b, os.path.join(TMPDIR, a))
    tex = tex.replace(b, a)

shutil.copyfile("vc.tex", os.path.join(TMPDIR, "vc.tex"))
open(os.path.join(TMPDIR, "ms.tex"), "w").write(tex)
check_call(" ".join(["cd", TMPDIR+";",
                     "tar", "-cf", os.path.join("..", outfn), "*"]),
           shell=True)
shutil.rmtree(TMPDIR)

print("Wrote file: '{0}'".format(outfn))
