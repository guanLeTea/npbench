# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for arc_distance's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# The port returns its result and `output_args` is empty, so `returns` must be declared. All
# four inputs are rank-1 pointers (no VLA extent states N), so N comes from theta_1's length --
# the same gap the module docstring notes for durbin. `temp` is the port's own intermediate.
ADAPTER = Adapter(
    symbols={"N": lambda a: int(np.shape(a["theta_1"])[0])},
    outputs=[("out", np.float64, lambda a: (a["N"], ), lambda shape, a: np.zeros(shape, np.float64)),
             ("temp", np.float64, lambda a: (a["N"], ), lambda shape, a: np.zeros(shape, np.float64))],
    returns=["out"],
)
