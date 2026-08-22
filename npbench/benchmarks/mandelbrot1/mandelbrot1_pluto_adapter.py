# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for mandelbrot1's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# The port takes only scalars, so every array is allocated here: `C` is its transient grid, `Z`
# and `N` are what it returns, in that order.
ADAPTER = Adapter(
    outputs=[
        ("C", np.complex128, lambda a: (a["YN"], a["XN"]), lambda shape, a: np.zeros(shape, np.complex128)),
        ("Z", np.complex128, lambda a: (a["YN"], a["XN"]), lambda shape, a: np.zeros(shape, np.complex128)),
        ("N", np.int64, lambda a: (a["YN"], a["XN"]), lambda shape, a: np.zeros(shape, np.int64)),
    ],
    returns=["Z", "N"],
)
