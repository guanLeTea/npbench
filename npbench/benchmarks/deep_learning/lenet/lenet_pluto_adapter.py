# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Argument adapter for lenet's Pluto scop; see pluto_framework.local_adapter."""

import numpy as np

from npbench.infrastructure.pluto_framework import Adapter

# H and W come from `input`'s own VLA extents (input is (N, H, W, 1)); every other symbol
# below is a size the LeNet-5 architecture derives from them but that no real array's shape
# states directly, so each is read off the SAME arithmetic `initialize()` uses to build
# `C_before_fc1` (deep_learning/lenet/lenet.py): two 5x5 valid convolutions (-4 each) and two
# 2x2 maxpools (//2 each). `fc1w` keeps its real (C_before_fc1, 120) shape and dtype -- the
# scop reinterprets that SAME contiguous buffer as a [PH2][PW2][16][120] VLA instead of
# materializing a separate flattened copy of the pooled activation; see the .c file's header.
_z32 = lambda shape, a: np.zeros(shape, np.float32)


def _h1(a):
    return int(a["H"]) - 4


def _w1(a):
    return int(a["W"]) - 4


def _ph1(a):
    return _h1(a) // 2


def _pw1(a):
    return _w1(a) // 2


def _h2(a):
    return _ph1(a) - 4


def _w2(a):
    return _pw1(a) - 4


def _ph2(a):
    return _h2(a) // 2


def _pw2(a):
    return _w2(a) // 2


ADAPTER = Adapter(
    symbols={
        "H1": _h1,
        "W1": _w1,
        "PH1": _ph1,
        "PW1": _pw1,
        "H2": _h2,
        "W2": _w2,
        "PH2": _ph2,
        "PW2": _pw2,
    },
    outputs=[
        ("conv1out", np.float32, lambda a: (a["N"], a["H1"], a["W1"], 6), _z32),
        ("pool1out", np.float32, lambda a: (a["N"], a["PH1"], a["PW1"], 6), _z32),
        ("conv2out", np.float32, lambda a: (a["N"], a["H2"], a["W2"], 16), _z32),
        ("pool2out", np.float32, lambda a: (a["N"], a["PH2"], a["PW2"], 16), _z32),
        ("x1", np.float32, lambda a: (a["N"], 120), _z32),
        ("x2", np.float32, lambda a: (a["N"], 84), _z32),
        ("out", np.float32, lambda a: (a["N"], 10), _z32),
    ],
    returns=["out"],
)
