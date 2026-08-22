# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for gesummv's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# y = alpha*A*x + beta*B*x. The port returns y; the scop writes it through a pointer and zeroes
# it itself. `tmp` is PolyBench/C's own scratch parameter of kernel_gesummv.
ADAPTER = Adapter(outputs=[("y", np.float64, lambda a: (a["N"], ), lambda shape, a: np.zeros(shape, np.float64)),
                           ("tmp", np.float64, lambda a: (a["N"], ), lambda shape, a: np.zeros(shape, np.float64))],
                  returns=["y"])
