# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for contour_integral's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# num_int_pts is int_pts's own VLA extent. P0/P1 are what the port returns; Tz and X are its
# per-contour-point transients, lifted to parameters.
_z = lambda shape, a: np.zeros(shape, np.complex128)

ADAPTER = Adapter(
    outputs=[
        ("P0", np.complex128, lambda a: (a["NR"], a["NM"]), _z),
        ("P1", np.complex128, lambda a: (a["NR"], a["NM"]), _z),
        ("Tz", np.complex128, lambda a: (a["NR"], a["NR"]), _z),
        ("X", np.complex128, lambda a: (a["NR"], a["NM"]), _z),
        ("zz", np.complex128, lambda a: (a["slab_per_bc"] + 1, ), _z),
    ],
    returns=["P0", "P1"],
)
