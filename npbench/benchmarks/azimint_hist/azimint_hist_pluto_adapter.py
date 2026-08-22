# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for azimint_hist's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# N comes from `data`/`radius`'s own VLA extent; `npt` is a real input arg. `histu` is
# scratch (the raw per-bin counts, never returned); `histw` starts as the weighted sums and
# ends as the port's own returned ratio, in place.
_z = lambda shape, a: np.zeros(shape, np.float64)

ADAPTER = Adapter(
    outputs=[
        ("histu", np.float64, lambda a: (a["npt"], ), _z),
        ("histw", np.float64, lambda a: (a["npt"], ), _z),
    ],
    returns=["histw"],
)
