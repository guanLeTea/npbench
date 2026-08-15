# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""DaCe CPU measured as ONE pipeline: ``auto_optimize`` and nothing else.

``dace_cpu`` builds three SDFG variants per kernel -- ``fusion``, ``parallel`` and ``auto_opt`` --
and records all three, leaving the choice of which to report to the reader. Taking the fastest of
them is a reasonable answer to "how fast is DaCe", but it is the wrong baseline for a comparison
against a single fixed optimizer: Pluto runs one pipeline, so scoring it against the best of three
DaCe transformations per kernel compares one optimizer against a per-kernel search.

This framework answers "how fast is DaCe's auto_optimize" instead, which is the comparison the
thesis makes. It is a NARROWING, not a reimplementation: the class body below is one attribute,
and every SDFG it produces comes from :class:`DaceFramework`'s own code path.

``auto_opt`` is derived from the simplified (``strict``) SDFG, never from the fused or
parallelized ones (``parallel`` is the only variant with such a dependency, and it derives from
``fusion``). Narrowing therefore skips the other two entirely without changing what
``auto_optimize`` is handed, and ``set_fast_implementations`` is already skipped for ``auto_opt``
in the shared path -- so this measures exactly the SDFG that ``dace_cpu`` labels ``auto_opt``,
at roughly a third of the build cost.

The DaCe version/branch/commit stamp is inherited unchanged, so rows written by this framework
carry the same ``2.0.0a5+<branch>@<commit>`` identification as ``dace_cpu``.
"""
from npbench.infrastructure.dace_framework import DaceFramework
from typing import Tuple


class DaceAutoOptFramework(DaceFramework):
    """``dace_cpu`` restricted to the ``auto_opt`` pipeline. See the module docstring."""

    #: The single pipeline this framework builds, times and records. Everything else --
    #: parsing, simplification, compilation, the results row -- is DaceFramework's.
    VARIANTS: Tuple[str, ...] = ("auto_opt", )
