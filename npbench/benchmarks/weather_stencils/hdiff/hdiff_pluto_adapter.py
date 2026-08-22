# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for hdiff's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# out_field is mutated in place (already `output_args`), so no `returns` is needed. lap_field,
# flx_field and fly_field are the port's own temporaries, shaped from its slice arithmetic.
_z = lambda shape, a: np.zeros(shape, np.float64)

ADAPTER = Adapter(outputs=[("lap_field", np.float64, lambda a: (a["I"] + 2, a["J"] + 2, a["K"]), _z),
                           ("flx_field", np.float64, lambda a: (a["I"] + 1, a["J"], a["K"]), _z),
                           ("fly_field", np.float64, lambda a: (a["I"], a["J"] + 1, a["K"]), _z)])
