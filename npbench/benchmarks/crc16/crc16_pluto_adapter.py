# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for crc16's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# The port returns a plain Python int; the scop writes it through a one-element `out`.
#
# `data` needs no entry here: it is a NumPy uint8 array (crc16.py's initializer:
# `rng.integers(0, 256, size=(N,), dtype=np.uint8)`), the scop declares it `uint8_t`, and
# `_ELEM_TYPES` now carries that type, so the actual argument's dtype already matches the
# scop's declared element type exactly and passes through with no conversion.
ADAPTER = Adapter(outputs=[("out", np.int32, lambda a: (1, ), lambda shape, a: np.zeros(shape, np.int32))],
                  returns=["out"])
