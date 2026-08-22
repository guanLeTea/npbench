# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for doitgen's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# The port writes A in place (`A[:] = ...`) and returns nothing; `output_args` names A, so no
# `returns` entry is needed. `sum` is PolyBench/C's own scratch parameter of kernel_doitgen.
ADAPTER = Adapter(
    outputs=[("sum", np.float64, lambda a: (a["NP"], ), lambda shape, a: np.zeros(shape, np.float64))])
