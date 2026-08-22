# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for covariance2's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# The port returns cov and subtracts the column means from `data` in place, exactly as the scop
# does. `mean` is PolyBench/C's own scratch parameter of kernel_covariance. covariance2 computes
# the same thing through np.cov(data.T): float_n is initialized to N in both ports, so np.cov's
# (N-1) denominator and its mean over N agree with the scop term for term.
ADAPTER = Adapter(outputs=[("cov", np.float64, lambda a: (a["M"], a["M"]), lambda shape, a: np.zeros(shape, np.float64)),
                           ("mean", np.float64, lambda a: (a["M"], ), lambda shape, a: np.zeros(shape, np.float64))],
                  returns=["cov"])
