# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for vadv's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# utens_stage is mutated in place (already `output_args`), so no `returns` is needed. ccol, dcol
# and data_col are the port's own Thomas-solver temporaries (allocated with `np.ndarray`,
# uninitialized, in the port); zero-filled here since every cell either side ever reads is
# written before it is read (see the header comment on ccol's unused last k-slice).
_z = lambda shape, a: np.zeros(shape, np.float64)

ADAPTER = Adapter(outputs=[("ccol", np.float64, lambda a: (a["I"], a["J"], a["K"]), _z),
                           ("dcol", np.float64, lambda a: (a["I"], a["J"], a["K"]), _z),
                           ("data_col", np.float64, lambda a: (a["I"], a["J"]), _z)])
