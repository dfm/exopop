#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gevent
from gevent import monkey
monkey.patch_all()

import os
import urllib2

url = "http://archive.stsci.edu/pub/kepler/catalogs/cdpp_quarter{0:d}.txt.gz"
outdir = "data"
try:
    os.makedirs(outdir)
except os.error:
    print("Output directory already exists.")


def download_quarter(n):
    print("Downloading quarter {0}".format(n))
    u = url.format(n)
    fn = os.path.split(u)[1]
    try:
        data = urllib2.urlopen(u).read()
    except Exception as e:
        print("Quarter {0} failed with error:".format(n))
        print(e)
    else:
        with open(os.path.join(outdir, fn), "wb") as f:
            f.write(data)


if __name__ == "__main__":
    jobs = [gevent.spawn(download_quarter, n) for n in range(0, 17)]
    gevent.joinall(jobs)
