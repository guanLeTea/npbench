# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for k2mm's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# D[:] = alpha * A @ B @ C + beta * D. The port mutates D in place and `output_args` is ["D"],
# so no `returns` is needed. `tmp` is PolyBench/C's own scratch parameter of kernel_2mm.
ADAPTER = Adapter(outputs=[("tmp", np.float64, lambda a: (a["NI"], a["NJ"]), lambda shape, a: np.zeros(
    shape, np.float64))])
