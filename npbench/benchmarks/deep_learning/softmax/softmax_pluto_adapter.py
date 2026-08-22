# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for softmax's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# float32 throughout, as the port is. The port returns the softmax; the scop writes it through
# `out`, and tmp_max/tmp_sum are the port's own keepdims temporaries.
_z32 = lambda shape, a: np.zeros(shape, np.float32)

ADAPTER = Adapter(outputs=[("out", np.float32, lambda a: (a["N"], a["H"], a["SM"], a["SM"]), _z32),
                           ("tmp_max", np.float32, lambda a: (a["N"], a["H"], a["SM"]), _z32),
                           ("tmp_sum", np.float32, lambda a: (a["N"], a["H"], a["SM"]), _z32)],
                  returns=["out"])
