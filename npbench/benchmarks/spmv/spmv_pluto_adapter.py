# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for spmv's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# Mp1 (A_row's own length), NNZ (A_col/A_val's own length) and XN (x's own length) all come
# from the three real arrays' VLA extents -- no `symbols` entry needed. The port returns `y`
# and `output_args` is empty, so `returns` must be declared; `y`'s length is A_row.size - 1,
# the port's own `np.zeros(A_row.size - 1, A_val.dtype)`.
ADAPTER = Adapter(
    outputs=[("y", np.float64, lambda a: (a["Mp1"] - 1, ), lambda shape, a: np.zeros(shape, np.float64))],
    returns=["y"],
)
