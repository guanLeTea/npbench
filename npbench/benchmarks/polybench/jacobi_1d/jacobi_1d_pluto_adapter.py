# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for jacobi_1d's Pluto scop; see pluto_framework.local_adapter."""

from npbench.infrastructure.pluto_framework import Adapter

# A and B are already NPBench's own arrays, written in place, and `output_args` names both, so
# no outputs/returns are needed here.
ADAPTER = Adapter()

# PolyBench sweeps `for (t = 0; t < TSTEPS; t++)` -> TSTEPS iterations.
# NPBench's port sweeps `for t in range(1, TSTEPS)` -> TSTEPS - 1 iterations.
# Passing TSTEPS-1 makes the C loop run the port's iteration count, exactly like jacobi_2d and
# heat_3d in pluto_framework.ARG_OVERRIDES.
ARG_OVERRIDES = {"TSTEPS": lambda a: a["TSTEPS"] - 1}
