# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for nbody's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# N and Nt both come straight through as real input_args, so no `symbols` entry is needed.
# `acc` and `corr` are the port's own carried/transient state, never returned. `KE`/`PE` are
# what the port returns (length Nt + 1, float64, built via `np.ndarray` -- zero-filled here
# since every cell is written exactly once by the scop before being read).
_z64 = lambda shape, a: np.zeros(shape, np.float64)

ADAPTER = Adapter(
    outputs=[
        ("acc", np.float64, lambda a: (a["N"], 3), _z64),
        ("corr", np.float64, lambda a: (3, ), _z64),
        ("KE", np.float64, lambda a: (a["Nt"] + 1, ), _z64),
        ("PE", np.float64, lambda a: (a["Nt"] + 1, ), _z64),
    ],
    returns=["KE", "PE"],
)
