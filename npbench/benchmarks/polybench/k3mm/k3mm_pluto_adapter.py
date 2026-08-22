# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for k3mm's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# The port returns A @ B @ C @ D with no in-place output (`output_args` is empty), so `returns`
# must be declared or validation would compare nothing and pass vacuously. `E` and `F` are
# PolyBench/C's own scratch parameters of kernel_3mm; `G` is the final result.
_z = lambda shape, a: np.zeros(shape, np.float64)

ADAPTER = Adapter(outputs=[("E", np.float64, lambda a: (a["NI"], a["NJ"]), _z),
                           ("F", np.float64, lambda a: (a["NJ"], a["NL"]), _z),
                           ("G", np.float64, lambda a: (a["NI"], a["NL"]), _z)],
                  returns=["G"])
