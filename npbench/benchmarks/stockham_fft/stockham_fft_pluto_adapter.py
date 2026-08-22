# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for stockham_fft's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# `dft_mat` and `tmp_twid` are the port's own transients, lifted to parameters. `y` is what the
# port returns. Declared so the scop is complete; whether it is ever reached depends on polycc.
ADAPTER = Adapter(
    outputs=[
        ("dft_mat", np.complex128, lambda a: (a["R"], a["R"]), lambda shape, a: np.zeros(shape, np.complex128)),
        ("tmp_twid", np.complex128, lambda a: (a["N"], ), lambda shape, a: np.zeros(shape, np.complex128)),
    ],
    returns=["y"],
)
