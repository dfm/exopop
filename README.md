Exoplanet population inference
==============================

This repository contains the code and text for the paper [Exoplanet population
inference and the abundance of Earth analogs from noisy, incomplete catalogs]()
by Daniel Foreman-Mackey, David W. Hogg, and Timothy D. Morton.

The code lives in the `code` directory and the LaTeX source code for the paper
is in `document`.

**Code**

The meat of the probabilistic model is implemented in `population.py`. Then
there are a set of scripts that you can use to generate the figures from the
paper. You should look at the docstrings for details but the summary is:

* `simulate.py` generates synthetic catalogs from known occurrence rate
  density functions,
* `main.py` does the MCMC analysis on either real or simulated catalogs, and
* `results.py` analyzes the results of the MCMC, makes some figures, and thins
  the chain to the published form.

Attribution
-----------

This code is associated with and written specifically for [our paper](). If you
make any use of it, please cite it:

```
@article{exopop,
   author = {{Foreman-Mackey}, D. and {Hogg}, D.~W. and {Morton}, T.~D.},
    title = {Exoplanet population inference and the abundance of
             Earth analogs from noisy, incomplete catalogs},
  journal = {ArXiv --- submitted to ApJ},
     year = 2014,
   eprint = {},
      doi = {}
}
```

License
-------

Copyright 2014 Daniel Foreman-Mackey

The code in this repository is made available under the terms of the MIT
License. For details, see the LICENSE file.
