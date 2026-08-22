# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for cavity_flow's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# u, v, p are already NPBench's own arrays, written in place, and `output_args` names all
# three, so the port returns nothing and no `returns` is needed. b, pn, un, vn are the port's
# own temporaries (np.zeros((ny, nx)) / np.zeros_like(p) / np.zeros_like(u) / np.zeros_like(v)):
# scratch the scop needs as caller-allocated parameters, matched to the port's own zero
# initializer so any cell neither side writes still compares equal.
_zeros = lambda shape, a: np.zeros(shape, np.float64)

ADAPTER = Adapter(outputs=[("b", np.float64, lambda a: (a["ny"], a["nx"]), _zeros),
                           ("pn", np.float64, lambda a: (a["ny"], a["nx"]), _zeros),
                           ("un", np.float64, lambda a: (a["ny"], a["nx"]), _zeros),
                           ("vn", np.float64, lambda a: (a["ny"], a["nx"]), _zeros)])
