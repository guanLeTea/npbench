# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for go_fast's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# The port returns a + trace; the scop writes it through `out`. `trace` is the port's own
# scalar reduction, passed as a one-element scratch parameter.
ADAPTER = Adapter(outputs=[("out", np.float64, lambda a: np.shape(a["a"]), lambda shape, a: np.zeros(shape, np.float64)),
                           ("trace", np.float64, lambda a: (1, ), lambda shape, a: np.zeros(shape, np.float64))],
                  returns=["out"])
