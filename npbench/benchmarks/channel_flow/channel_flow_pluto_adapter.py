# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for channel_flow's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# u, v, p are already NPBench's own arrays, written in place, and `output_args` names all
# three. The port ALSO returns `stepcount`, so `returns` must name it or `Test._execute`'s
# zip of [stepcount, u, v, p] (numpy) against [u, v, p] (this column) would misalign every
# comparison by one slot. b, pn, un, vn are the port's own temporaries (np.zeros_like(u) /
# np.zeros_like(p) / np.zeros_like(u) / np.zeros_like(v)): scratch the scop needs as
# caller-allocated parameters. `stepcount` has no array counterpart in the port -- it is
# returned through a one-element int64 buffer because the exported symbol is `void`.
_zeros = lambda shape, a: np.zeros(shape, np.float64)

ADAPTER = Adapter(outputs=[("b", np.float64, lambda a: (a["ny"], a["nx"]), _zeros),
                           ("pn", np.float64, lambda a: (a["ny"], a["nx"]), _zeros),
                           ("un", np.float64, lambda a: (a["ny"], a["nx"]), _zeros),
                           ("vn", np.float64, lambda a: (a["ny"], a["nx"]), _zeros),
                           ("stepcount", np.int64, lambda a: (1, ), lambda shape, a: np.zeros(shape, np.int64))],
                  returns=["stepcount"])
