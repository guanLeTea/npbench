# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for azimint_naive's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# The port returns res; the scop writes it through a pointer. NPT is the port's own `npt`
# argument -- no array extent names it, so it comes straight from the actual. `sum`, `cnt` and
# `rmax` are the port's own scalar/per-bin temporaries, passed as scratch parameters.
ADAPTER = Adapter(symbols={"NPT": lambda a: int(a["npt"])},
                  outputs=[("res", np.float64, lambda a: (a["NPT"], ), lambda shape, a: np.zeros(shape, np.float64)),
                           ("sum", np.float64, lambda a: (a["NPT"], ), lambda shape, a: np.zeros(shape, np.float64)),
                           ("cnt", np.float64, lambda a: (a["NPT"], ), lambda shape, a: np.zeros(shape, np.float64)),
                           ("rmax", np.float64, lambda a: (1, ), lambda shape, a: np.zeros(shape, np.float64))],
                  returns=["res"])
