# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for mlp's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# float32 throughout, as the port is. The port returns the softmax output; the scop writes it
# through `out`. x1/x2 are the port's two hidden activations and tmp_max/tmp_sum are softmax's
# own keepdims temporaries -- all scratch, allocated here and never returned.
_z32 = lambda shape, a: np.zeros(shape, np.float32)

ADAPTER = Adapter(outputs=[("out", np.float32, lambda a: (a["N"], a["S2"]), _z32),
                           ("x1", np.float32, lambda a: (a["N"], a["S0"]), _z32),
                           ("x2", np.float32, lambda a: (a["N"], a["S1"]), _z32),
                           ("tmp_max", np.float32, lambda a: (a["N"], ), _z32),
                           ("tmp_sum", np.float32, lambda a: (a["N"], ), _z32)],
                  returns=["out"])
