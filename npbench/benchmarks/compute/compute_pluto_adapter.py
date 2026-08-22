# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for compute's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# The port returns its result and `output_args` is empty, so `returns` must be declared or
# validation would compare nothing and pass vacuously. array_1/array_2 are int64, so `out` is too.
ADAPTER = Adapter(outputs=[("out", np.int64, lambda a: (a["M"], a["N"]), lambda shape, a: np.zeros(
    shape, np.int64))],
                  returns=["out"])
