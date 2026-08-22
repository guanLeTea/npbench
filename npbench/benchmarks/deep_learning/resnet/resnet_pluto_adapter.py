# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for resnet's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# N, H, W, C1, C2 all come from the four VLA extents already declared on input/conv1/conv2/conv3
# (no `symbols` needed). Every entry below is scratch: the port allocates none of these and
# returns only the final block output, `out`.
_z64 = lambda shape, a: np.zeros(shape, np.float64)
_z32 = lambda shape, a: np.zeros(shape, np.float32)

ADAPTER = Adapter(
    outputs=[
        ("padded", np.float64, lambda a: (a["N"], a["H"] + 2, a["W"] + 2, a["C2"]), _z64),
        ("mean1", np.float64, lambda a: (a["H"] + 2, a["W"] + 2, a["C2"]), _z64),
        ("std1", np.float64, lambda a: (a["H"] + 2, a["W"] + 2, a["C2"]), _z64),
        ("buf2", np.float32, lambda a: (a["N"], a["H"], a["W"], a["C2"]), _z32),
        ("acc2", np.float64, lambda a: (a["N"], a["H"], a["W"], a["C2"]), _z64),
        ("mean2", np.float32, lambda a: (a["H"], a["W"], a["C2"]), _z32),
        ("std2", np.float32, lambda a: (a["H"], a["W"], a["C2"]), _z32),
        ("out", np.float32, lambda a: (a["N"], a["H"], a["W"], a["C1"]), _z32),
        ("mean3", np.float32, lambda a: (a["H"], a["W"], a["C1"]), _z32),
        ("std3", np.float32, lambda a: (a["H"], a["W"], a["C1"]), _z32),
    ],
    returns=["out"],
)
