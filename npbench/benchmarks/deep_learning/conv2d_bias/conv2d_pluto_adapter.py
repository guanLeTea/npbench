# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for conv2d_bias's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# float32 throughout, as the port is. H_out/W_out are derived extents with no array of their
# own, resolved here from H, W, K (already resolved from input's and weights' own shapes) before
# `output` is sized. The port returns its result and `output_args` is empty, so `returns` names
# it explicitly.
ADAPTER = Adapter(
    symbols={
        "H_out": lambda a: int(a["H"]) - int(a["K"]) + 1,
        "W_out": lambda a: int(a["W"]) - int(a["K"]) + 1,
    },
    outputs=[("output", np.float32, lambda a: (a["N"], a["H_out"], a["W_out"], a["C_out"]),
             lambda shape, a: np.zeros(shape, np.float32))],
    returns=["output"],
)
