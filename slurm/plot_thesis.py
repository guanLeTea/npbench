#!/usr/bin/env python
"""Thesis report for a PolyBench-derived NPBench campaign: two pages, plus an
optional third when per-sample statistics are supplied.

Page 1 -- a compact speedup heatmap in the style of the HPCAgent-Bench figure: one row per
kernel, columns ``Pluto | DaCe auto_opt | NumPy``, kernels grouped by PolyBench category. Colour
is a diverging RdYlGn ramp on ``log10(speedup)`` centred at 1x, so "faster than NumPy" trends
green, "slower" trends red and 1x is neutral. The log is what makes 0.5x read as far from centre
as 2x; a linear ramp would squeeze every regression into a narrow band next to neutral.

Page 2 -- the diagnostics behind page 1: per kernel, the NumPy runtime, both speedups or their
status, the validation verdict, and the reason a cell is blank. Page 1 is the comparison; page 2
is why some of it is missing.

Page 3 (with ``--stats``) -- raw runtime distributions for a couple of contrasting kernels, as
violins over every sample, with the median and its bootstrap confidence interval. ``--extra-violins``
appends further such pages for the kernels in :data:`EXTRA_VIOLIN_PAGES`.

A speedup is drawn ONLY from a row whose ``validated`` flag is set. A declined, crashed or
unvalidated cell is hatched and labelled, never coloured on the speedup ramp -- the two encode
different things and sharing a scale would let a missing measurement read as a slow one.
"""
import argparse
import json
import math
import pathlib
import sqlite3
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from npbench.infrastructure.pluto_framework import POLYCC_ARGS
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

#: PolyBench/C 4.2.1 category per kernel, in the order the suite itself groups them. Drives the
#: row order and the brackets down the right-hand edge.
CATEGORY = [
    ("linear-algebra / blas", ["gemm", "gemver", "syrk", "syr2k", "trmm"]),
    ("linear-algebra / kernels", ["atax", "bicg", "mvt"]),
    ("linear-algebra / solvers", ["cholesky", "durbin", "gramschmidt", "lu", "ludcmp", "trisolv"]),
    ("datamining", ["correlation"]),
    ("medley", ["deriche", "floyd_warshall", "nussinov"]),
    ("stencils", ["adi", "fdtd_2d", "heat_3d", "jacobi_2d", "seidel_2d"]),
]

#: Speedup range the colour ramp spans, as a factor either side of 1x. 32x saturates the ends;
#: beyond that the exact magnitude is carried by the printed number, not the colour.
RAMP = 32.0

#: The NumPy column is a runtime, not a speedup, so it gets its own flat colour rather than a
#: point on the diverging ramp -- it is the reference the ramp is measured against.
NUMPY_FILL = "#fdf6d0"
#: A cell with no measurement. Grey and hatched: distinct from every colour on the ramp.
GAP_FILL = "#e4e7e9"
GAP_EDGE = "#9aa5ab"

#: Lines a page-2 note may occupy. Four 100-character lines at 0.26 row-units of leading fit
#: inside one row's height without touching its neighbours; a fifth would overlap.
_NOTE_LINES = 4

STATUS_SHORT = {
    "declined": "declined",
    "error": "crashed",
    "invalid": "invalid",
    "missing": "n/a",
}


def diverging():
    """RdYlGn, but with the neutral midpoint lightened so 1x reads as "no change" not "warning"."""
    return LinearSegmentedColormap.from_list(
        "speedup", ["#8c1620", "#c94741", "#f08b60", "#fbd48a", "#f2f2ef", "#bfe08f", "#71c05a", "#2f9e46", "#166b31"])


def speedup_text(x):
    """``42.4x`` / ``2.1x`` / ``0.7x`` -- plain factors, as the report asks."""
    if x is None or not np.isfinite(x):
        return ""
    if x >= 100:
        return "%dx" % round(x)
    if x >= 10:
        return "%.1fx" % x
    return "%.2fx" % x if x < 1 else "%.1fx" % x


def runtime_text(t):
    if t is None or not np.isfinite(t):
        return ""
    if t < 1e-3:
        return "%.0f us" % (t * 1e6)
    if t < 1.0:
        return "%.1f ms" % (t * 1e3)
    return "%.2f s" % t


def run_kind(repeat):
    """How the run should describe itself, given how many repetitions it has.

    REPEAT=1 is a correctness check whose timings are single samples and must say so. A repeated
    run reports medians and must NOT carry the verification wording, which would understate
    numbers that are fit to quote.
    """
    if repeat <= 1:
        return "VERIFICATION RUN (single sample per kernel), not a performance measurement"
    return "medians over %d repetitions per kernel" % repeat


def caption_lines(subcaption):
    """``subcaption`` as one line, or split at its separator when it is too long for the page.

    Page 1 is 7 inches wide and centres this text, so an over-long line runs off BOTH margins --
    it loses the start of the DaCe stamp and the end of the clang flags at once, which is how it
    lost "-march=native" when the Clan frontend was named. Splitting on the "|" the caption
    already uses keeps each half on its own line without inventing a break point.
    """
    if len(subcaption) <= 118 or "|" not in subcaption:
        return [subcaption]
    return [s.strip() for s in subcaption.split("|", 1)]


def short_reason(status, reason):
    """Two-to-three sentences: what fails, at which stage, and the concrete evidence.

    A bare label like "dropped statements" is not an explanation -- it names a symptom without
    saying which component lost them or why. Each cause below states the stage (PET extraction /
    Pluto transformation / generated C / run time / semantic incompatibility) and the measured
    fact behind the attribution.
    """
    if not reason:
        return STATUS_SHORT.get(status, status)
    r = " ".join(reason.split())

    # A kernel with no tracked scop was never offered to polycc. Across a corpus with no Pluto
    # references at all this reason repeats on every row, so it is stated compactly and, above
    # all, as UNSUPPORTED rather than as a Pluto failure.
    if "no tracked PolyBench scop" in r:
        return ("unsupported here -- no PolyBench scop is tracked for this kernel yet, so polycc "
                "was never invoked. This is a gap in our benchmark, not a Pluto failure.")

    if "dropped the statements writing" in r:
        names = r.split("dropped the statements writing", 1)[1].split(" in ", 1)[0]
        names = [n.strip() for n in names.replace("`", "").split(",") if n.strip()]
        shown = ", ".join(names[:4]) + (", ..." if len(names) > 4 else "")
        return ("Pluto's statement list omits the statements writing the scop-local scalar "
                "temporaries (%s). PET does extract them -- they appear in pet's own scop dump -- "
                "so they are lost when Pluto consumes that scop, not during extraction. The "
                "emitted C reads them uninitialised and returns NaN." % shown)

    if "integer literals too large for int64" in r:
        return ("Generated C: Pluto emitted a loop guard whose coefficient exceeds int64, so the "
                "guard is undefined at run time and silently skips its loop. Fixed for the "
                "recovered kernels by --codegen-context=1; this kernel still trips it.")

    if "differ semantically" in r and "b = 1.0 + mul2" in r:
        return ("Semantic incompatibility, not a Pluto failure: the port sets b = 1.0 + mul2 where "
                "PolyBench/C 4.2.1 sets b = 1.0 + mul1, so the two solve different tridiagonal "
                "systems (81 vs 161 at preset S). Present since NPBench's first commit and identical "
                "in all ten adi implementations -- an upstream porting bug, but correcting it would "
                "change what the suite measures.")

    if "differ semantically" in r and "normalization" in r:
        return ("Semantic incompatibility, not a Pluto failure: the port's constant k has "
                "denominator 1.0 + alpha*exp(-alpha) - exp(2*alpha) where PolyBench/C 4.2.1 has "
                "1.0 + 2.0*alpha*exp(-alpha); the factor 2.0 is missing. k scales a1..a8, so every "
                "pixel is scaled -- relative error 2.07. Reproduced on the untransformed reference, "
                "so nothing Pluto did is involved.")

    if "parallelization is unsound" in r:
        return ("Generated C: the transformation is correct but its parallel decoration is not. "
                "polycc marks the tiled i loop omp parallel for while the kernel reads row k+1 of "
                "table as another thread writes it. At N=200: 0 of 20 runs differ on one thread, "
                "20 of 20 on eight and on 32. It is the one failure that can pass validation by "
                "luck, so it is refused up front.")

    if "killed by signal" in r:
        return ("Run time: polycc transforms and clang compiles cleanly, but the transformed "
                "binary segfaults on its first call, taking the benchmark process with it "
                "(exit 139). Reproduces on unmodified PolyBench/C 4.2.1, so it is Pluto's "
                "generated code, not our ABI adaptation.")

    if "polycc failed" in r and "Clan" in r:
        return ("Two defects. Clan transposes adi's scop parameters: its domain for v[0][i] says -t+N>=0 and "
                "-i+TSTEPS-2>=0 where the loops are t=1..TSTEPS and i=1..N-1, so polycc guards on TSTEPS and "
                "can write past the arrays. Hoisting the scalar setup fixes that, but polycc then still emits "
                "the back-substitution before the loop producing its p and q.")

    if "polycc failed" in r:
        return ("Pluto transformation: polycc aborts inside pluto_auto_transform on the assertion "
                "hyp_search_mode == LAZY || num_sols_left == num_ind_sols_req - num_ind_sols_found. "
                "It fails before any C is generated, and reproduces on unmodified PolyBench/C 4.2.1.")

    if "did not finish within" in r:
        return "Pluto transformation: polycc did not terminate within its timeout."
    if "marked no loop parallel" in r:
        return "Pluto marked no loop parallel, so the column would time a sequential build."
    return r


def load(db, preset):
    conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    rows = conn.execute(
        "select benchmark, framework, details, validated, time from results where preset = ?",
        (preset, )).fetchall()
    stamps = [v for (v, ) in conn.execute("select distinct version from results")]
    conn.close()
    acc = {}
    for bench, fw, det, valid, t in rows:
        acc.setdefault((bench, fw), {}).setdefault((det or "", bool(valid)), []).append(t)
    out = {k: {kk: float(np.median(v)) for kk, v in var.items()} for k, var in acc.items()}
    return out, stamps


def pick(variants, pin=None):
    ok = [(t, name) for (name, valid), t in variants.items() if valid and (pin is None or name == pin)]
    return min(ok) if ok else (None, None)


def build_rows(args):
    data, stamps = load(args.db, args.preset)
    status = json.load(open(args.status)) if args.status and pathlib.Path(args.status).is_file() else {}
    repo = pathlib.Path(__file__).resolve().parent.parent

    def dbk(k):
        try:
            return json.loads((repo / "bench_info" / ("%s.json" % k)).read_text())["benchmark"]["short_name"]
        except Exception:
            return k

    # The grouping is PolyBench's by default. `--categories` supplies a different one for a
    # corpus PolyBench's categories do not describe (e.g. the wider NPBench kernels), as a JSON
    # [[label, [kernel, ...]], ...]. Without it nothing changes.
    categories = CATEGORY
    if getattr(args, "categories", None) and pathlib.Path(args.categories).is_file():
        categories = [(lab, list(ks)) for lab, ks in json.loads(pathlib.Path(args.categories).read_text())]

    known = [k for _, ks in categories for k in ks]
    if args.kernels and pathlib.Path(args.kernels).is_file():
        wanted = [l.strip() for l in open(args.kernels) if l.strip()]
    else:
        wanted = known
    groups = []
    for label, ks in categories:
        present = [k for k in ks if k in wanted]
        if present:
            groups.append((label, present))
    # Anything the category table does not know about still gets shown rather than dropped.
    leftover = [k for k in wanted if k not in known]
    if leftover:
        groups.append(("other", leftover))

    rows = []
    for label, ks in groups:
        for k in ks:
            b = dbk(k)
            nt, _ = pick(data.get((b, "numpy"), {}))
            e = {"kernel": k, "group": label, "numpy": nt}
            for role, fw, pin in (("pluto", "pluto", None),
                                  ("dace", args.dace_framework, args.dace_variant)):
                var = data.get((b, fw), {})
                t, variant = pick(var, pin)
                st = (status.get(k, {}) or {}).get(fw, {})
                if t is not None and nt:
                    e[role] = {"status": "validated", "speedup": nt / t, "time": t, "variant": variant,
                               # a validated measurement can still be single-threaded; see page 2
                               "sequential": bool(st.get("sequential")),
                               "note": st.get("note", "")}
                elif var:
                    e[role] = {"status": "invalid",
                               "reason": ("variant %r not validated" % pin) if pin else
                               st.get("reason", "ran but did not validate")}
                else:
                    e[role] = {"status": st.get("status", "missing"),
                               "reason": st.get("reason", "no result recorded")}
            rows.append(e)
    return rows, groups, stamps


# --------------------------------------------------------------------------------------- page 1

def page_overview(rows, groups, args, cmap, norm):
    n = len(rows)
    # Thesis-page width; height scales with the row count and keeps cells near-square.
    fig_h = 2.00 + 0.285 * (n + len(groups) + 1)
    fig = plt.figure(figsize=(7.0, fig_h))
    ax = fig.add_axes([0.215, 0.100, 0.50, 0.775])

    cols = ["Pluto", args.dace_column_label, "NumPy"]
    ax.set_xlim(0, 3)
    ax.set_ylim(0, n + 1)     # +1 for the aggregate row at the top
    ax.invert_yaxis()
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(cols, fontsize=8.5)
    ax.set_yticks([])
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=6)
    for s in ax.spines.values():
        s.set_visible(False)

    def cell(x, y, facecolor, text, textcolor="black", hatch=None, weight="normal", size=7.4):
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=facecolor, edgecolor="white", lw=1.1,
                               hatch=hatch, zorder=2))
        if text:
            ax.text(x + 0.5, y + 0.5, text, ha="center", va="center", fontsize=size,
                    color=textcolor, zorder=3, fontweight=weight)

    def fg(v):
        """Dark text on the pale middle of the ramp, white on the saturated ends."""
        return "white" if abs(math.log10(v)) > math.log10(RAMP) * 0.55 else "#20262b"

    # Aggregate row: geometric mean over the kernels each column actually answered. Stated as
    # such in the caption, because the two columns are means over DIFFERENT kernel sets.
    yl = []
    for j, role in enumerate(("pluto", "dace")):
        sp = [r[role]["speedup"] for r in rows if r[role]["status"] == "validated"]
        if sp:
            g = float(np.exp(np.mean(np.log(sp))))
            cell(j, 0, cmap(norm(math.log10(g))), speedup_text(g), fg(g), weight="bold", size=8.0)
            yl.append((role, g, len(sp)))
        else:
            cell(j, 0, GAP_FILL, "", hatch="///")
    # The two column means are over different kernel sets (Pluto answers fewer), so the
    # like-for-like pair over the kernels BOTH answered is stated in the footnote rather than
    # leaving the header row to imply a comparison it does not make.
    shared = [r for r in rows if r["pluto"]["status"] == "validated" and r["dace"]["status"] == "validated"]
    gshared = {}
    for role in ("pluto", "dace"):
        sp = [r[role]["speedup"] for r in shared]
        if sp:
            gshared[role] = float(np.exp(np.mean(np.log(sp))))
    tot = [r["numpy"] for r in rows if r["numpy"]]
    cell(2, 0, NUMPY_FILL, runtime_text(float(np.exp(np.mean(np.log(tot))))) if tot else "",
         "#20262b", weight="bold", size=8.0)
    ax.text(-0.12, 0.5, "geo-mean", ha="right", va="center", fontsize=8.2, fontweight="bold")

    for i, r in enumerate(rows):
        y = i + 1
        for j, role in enumerate(("pluto", "dace")):
            e = r[role]
            if e["status"] == "validated":
                v = e["speedup"]
                cell(j, y, cmap(norm(math.log10(v))), speedup_text(v), fg(v))
            else:
                cell(j, y, GAP_FILL, STATUS_SHORT.get(e["status"], e["status"]), "#5b666d",
                     hatch="///", size=6.4)
        cell(2, y, NUMPY_FILL, runtime_text(r["numpy"]), "#20262b")
        ksize = 8.0 if len(r["kernel"]) <= 17 else 8.0 * 17.0 / len(r["kernel"])
        ax.text(-0.12, y + 0.5, r["kernel"], ha="right", va="center", fontsize=max(ksize, 5.8))

    # Category brackets down the right edge.
    top = 1
    for label, ks in groups:
        lo, hi = top, top + len(ks)
        ax.plot([3.06, 3.14, 3.14, 3.06], [lo + 0.08, lo + 0.08, hi - 0.08, hi - 0.08],
                color="#7b878e", lw=0.9, clip_on=False)
        ax.text(3.20, (lo + hi) / 2, label, rotation=270, va="center", ha="left", fontsize=6.6,
                color="#414a50")
        if top > 1:
            ax.plot([-0.02, 3.0], [lo, lo], color="#9aa5ab", lw=1.0, clip_on=False, zorder=4)
        top = hi

    if len(gshared) == 2:
        fig.text(0.5, 0.012,
                 "geo-mean row is over the kernels each column answered (Pluto %d, DaCe %d). "
                 "Over the %d kernels BOTH answered: Pluto %s, DaCe %s."
                 % (sum(1 for r in rows if r["pluto"]["status"] == "validated"),
                    sum(1 for r in rows if r["dace"]["status"] == "validated"),
                    len(shared), speedup_text(gshared["pluto"]), speedup_text(gshared["dace"])),
                 ha="center", va="bottom", fontsize=6.6, color="#455055")

    fig.text(0.5, 0.985, args.title, ha="center", va="top", fontsize=11.5, fontweight="bold")
    fig.text(0.5, 0.963,
             "preset %s, REPEAT=%d -- %s"
             % (args.preset, args.repeat, args.run_label),
             ha="center", va="top", fontsize=7.6, color="#455055")
    for _i, _line in enumerate(caption_lines(args.subcaption)):
        fig.text(0.5, 0.947 - 0.0115 * _i, _line, ha="center", va="top",
                 fontsize=7.0, color="#455055")

    # Colour bar: ticks are factors, not log units, so the axis reads in the same notation as the
    # cells.
    cax = fig.add_axes([0.215, 0.060, 0.50, 0.012])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    ticks = [1 / RAMP, 0.1, 0.5, 1, 2, 10, RAMP]
    cb.set_ticks([math.log10(t) for t in ticks])
    cb.set_ticklabels([("%gx" % t) if t >= 1 else ("%.2gx" % t) for t in ticks])
    cb.ax.tick_params(labelsize=6.4, length=2, pad=1.5)
    cb.outline.set_visible(False)
    cb.set_label("speedup vs NumPy   (green = faster, red = slower; log scale)      |      hatched grey = no valid measurement, see page 2",
                 fontsize=6.6, labelpad=3)

    return fig, yl


# --------------------------------------------------------------------------------------- page 2

def page_details(rows, args):
    n = len(rows)
    fig_h = 1.35 + 0.42 * n
    fig = plt.figure(figsize=(10.0, fig_h))
    ax = fig.add_axes([0.035, 0.03, 0.945, 0.885])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 1)
    ax.invert_yaxis()
    ax.axis("off")

    # The kernel column is wide enough for the longest NPBench name at a legible size;
    # PolyBench's names all fit well inside it.
    xs = [0.0, 0.125, 0.245, 0.325, 0.443, 0.508]
    heads = ["kernel", "NumPy runtime", "Pluto", "DaCe auto_opt", "validated", "note / reason a cell is blank"]
    for x, h in zip(xs, heads):
        ax.text(x, 0.55, h, fontsize=7.8, fontweight="bold", va="center")
    ax.plot([0, 1], [0.95, 0.95], color="#5b666d", lw=1.0)

    for i, r in enumerate(rows):
        y = i + 1.35
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, y - 0.48), 1, 0.96, facecolor="#f4f6f7", edgecolor="none", zorder=0))
        p, d = r["pluto"], r["dace"]
        # `scattering_self_energies` is 24 characters and overruns the NumPy column at 7.8pt.
        # Scale only the names that would collide; every PolyBench name is short enough to be
        # unaffected.
        ksize = 7.8 if len(r["kernel"]) <= 19 else 7.8 * 19.0 / len(r["kernel"])
        ax.text(xs[0], y, r["kernel"], fontsize=max(ksize, 5.9), va="center", zorder=2)
        ax.text(xs[1], y, runtime_text(r["numpy"]), fontsize=7.8, va="center", zorder=2)

        for x, e in ((xs[2], p), (xs[3], d)):
            if e["status"] == "validated":
                ax.text(x, y, speedup_text(e["speedup"]), fontsize=7.8, va="center", zorder=2)
            else:
                ax.text(x, y, STATUS_SHORT.get(e["status"], e["status"]), fontsize=7.4,
                        va="center", color="#8c1620", zorder=2)

        both_ok = p["status"] == "validated" and d["status"] == "validated"
        ax.text(xs[4], y, "yes" if both_ok else ("DaCe only" if d["status"] == "validated" else "no"),
                fontsize=7.6, va="center", color="#2f7a3f" if both_ok else "#8a5a00", zorder=2)

        # Every column that did not validate gets its reason. Reporting only the first hides the
        # informative one whenever both fail -- on a corpus where Pluto declines uniformly, the
        # DaCe diagnostic is the entire content of the row.
        parts = []
        for role, e in (("Pluto", p), ("DaCe", d)):
            if e["status"] != "validated":
                parts.append("%s: %s" % (role, short_reason(e["status"], e.get("reason", ""))))
        note = "  ".join(parts)
        # A validated row can still need a qualifier. Pluto tiles `durbin` and `ludcmp`
        # correctly but finds no parallelism in either -- both are inherently sequential
        # recurrences -- so their speedups are one-thread numbers standing beside 72-thread
        # ones, and the table has to say so where the number is read.
        if not note and p.get("sequential"):
            if "unsound" in (p.get("note") or "") or "parallel for" in (p.get("note") or ""):
                note = ("Pluto: SINGLE-THREADED. polycc's transformation is correct and tiled, but "
                        "the `omp parallel for` it puts on the i loop is not: that loop reads rows "
                        "below i which later iterations write. The transform is used exactly as "
                        "generated and compiled without -fopenmp, so the pragma is inert. Validated "
                        "27/27 over three presets and 1/8/72 threads.")
            else:
                note = ("Pluto: correct and tiled, but polycc marked no loop parallel -- this is a "
                        "SINGLE-THREADED result. The kernel is an inherently sequential recurrence "
                        "and both polycc frontends agree there is no parallelism to find, so the "
                        "speedup is not comparable to the 72-thread rows above.")
        if note:
            # Centred on the row and adaptive: a truncated explanation is worse than a dense
            # one, since the whole point of this page is that the reason is complete.
            wrapped = textwrap.wrap(note, width=100)
            if len(wrapped) > _NOTE_LINES:
                # Silent truncation is the one failure this page cannot afford: it turns a complete
                # explanation into a sentence that stops mid-clause. Fail loudly instead, so the
                # text gets shortened rather than quietly cut.
                raise SystemExit(
                    "plot_thesis: the note for %r needs %d lines but only %d fit between rows.\n"
                    "Shorten it in short_reason() (budget is about %d characters).\n  %s"
                    % (r["kernel"], len(wrapped), _NOTE_LINES, _NOTE_LINES * 100, note))
            start = -0.26 * (len(wrapped) - 1) / 2.0
            for li, line in enumerate(wrapped):
                ax.text(xs[5], y + start + li * 0.26, line, fontsize=6.6, va="center",
                        color="#33393e", zorder=2)
        else:
            ax.text(xs[5], y, "both columns validated against the NumPy reference", fontsize=6.9,
                    va="center", color="#6b7379", zorder=2)

    fig.text(0.5, 0.982, "%s -- detailed results and diagnostics" % args.title,
             ha="center", va="top", fontsize=11.5, fontweight="bold")
    # Wrapped, not one line: page 2 is wider than page 1 but a long --run-label still overruns
    # both margins, which silently eats the start and the end of the sentence.
    sub = ("preset %s, REPEAT=%d, %s. Explains every blank cell on page 1; "
           "a speedup is shown only where the result validated."
           % (args.preset, args.repeat, args.run_label))
    sub_lines = textwrap.wrap(sub, width=150)
    for _i, _line in enumerate(sub_lines):
        fig.text(0.5, 0.958 - 0.0125 * _i, _line, ha="center", va="top",
                 fontsize=7.6, color="#455055")
    _y = 0.937 - 0.0125 * (len(sub_lines) - 1)
    for _i, _line in enumerate(caption_lines(args.subcaption)):
        fig.text(0.5, _y - 0.0115 * _i, _line, ha="center", va="top",
                 fontsize=7.0, color="#455055")
    return fig



# ------------------------------------------------------------------------------- page 3

#: Kernels given a distribution page, with the reason each was chosen. Two, deliberately: the
#: point is to show two OPPOSITE situations at readable size, not to reprint the whole suite.
VIOLIN_KERNELS = [
    ("gemm", "NumPy and DaCe reach a tuned GEMM; Pluto optimises the loop nest it was given "
             "rather than substituting a BLAS call"),
    ("heat_3d", "a regular affine stencil -- the case polyhedral tiling and parallelisation "
                "are designed for"),
]

#: Further distribution pages, grouped so no page carries more than three kernels. These are the
#: representative cases behind the heatmap rather than another pair of extremes: the rationale
#: strings stay descriptive of the KERNEL, never of the result, so the figure does not tell the
#: reader what to conclude from it.
EXTRA_VIOLIN_PAGES = [
    [("trmm", "triangular matrix multiply -- a dense BLAS-3 kernel with a triangular iteration "
              "space"),
     ("cholesky", "in-place Cholesky factorisation: a sequential outer loop over columns with "
                  "square-root and division on the diagonal"),
     ("gramschmidt", "modified Gram-Schmidt QR -- column-by-column orthogonalisation, each column "
                     "depending on all previous ones")],
    [("lu", "LU decomposition without pivoting, a triangular dependence structure over the whole "
            "matrix"),
     ("seidel_2d", "Gauss-Seidel 2-D stencil: an in-place sweep, so every point depends on "
                   "neighbours already updated in the same sweep"),
     ("mvt", "two independent matrix-vector products -- memory-bound BLAS-2, little arithmetic "
             "per byte moved")],
]


def raw_samples(db, preset, db_name, framework, variant=None):
    """Every VALIDATED runtime sample for one (kernel, framework), in seconds."""
    conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    rows = conn.execute(
        "select details, validated, time from results where preset = ? and benchmark = ? "
        "and framework = ?", (preset, db_name, framework)).fetchall()
    conn.close()
    return np.asarray([float(t) for d, v, t in rows
                       if v and (variant is None or not framework.startswith("dace") or (d or "") == variant)],
                      dtype=float)


def _unit(medians):
    """A common unit for one kernel's three violins, so they can be read against each other.

    Chosen from the SMALLEST of the three medians, not from the pooled samples. Pooling picks the
    unit off whichever implementation happens to sit in the middle, which made ``lu`` render in
    seconds -- printing its Pluto median as 0.09765 s -- while ``heat_3d``, whose middle series
    is just under a second, rendered in ms. Keying on the fastest series keeps the tightest
    distribution legible and makes the choice the same for every kernel here.
    """
    m = float(np.min(medians))
    if m >= 1e-3 and float(np.max(medians)) / 1e-3 > 99999.0:
        # ... unless that would print the slowest series with six digits; then fall back
        m = float(np.median(medians))
    if m < 1e-3:
        return 1e6, "us"
    if m < 1.0:
        return 1e3, "ms"
    return 1.0, "s"


def page_distributions(args, stats_by_pair, kernels=None):
    """Runtime distributions for :data:`VIOLIN_KERNELS`: one panel per (kernel, implementation).

    Raw runtimes, not speedups -- a speedup distribution folds two samples together and hides
    which side is actually spread out.

    Each implementation gets its OWN y-range, and that is the whole design decision. Within one
    kernel the three implementations differ by up to 170x while each distribution is tighter than
    3% of its own median, so a shared axis renders all three as flat lines and shows nothing; that
    is exactly what the first version of this page did. Units stay identical across a kernel's
    three panels (printed on each), and the medians are printed as text so the cross-implementation
    comparison is still readable -- it is simply carried by the numbers rather than by pixel height,
    which at these ratios is the only honest option.
    """
    cols = [("numpy", "NumPy", "#b8912a"), ("pluto", "Pluto", "#2f7a3f"),
            (args.dace_framework, "DaCe auto_opt", "#2b6ca3")]
    kernels = kernels if kernels is not None else VIOLIN_KERNELS
    nk = len(kernels)

    # Laid out in INCHES and converted, not in hard-coded figure fractions. The original two-kernel
    # page used fixed fractions, which silently break as soon as a page carries three kernels --
    # the third row lands at a negative coordinate and disappears off the bottom.
    head_in, row_in, foot_in = 1.15, 2.95, 1.00
    fig_h = head_in + row_in * nk + foot_in
    fig = plt.figure(figsize=(10.0, fig_h))
    head_f, row_f, foot_f = head_in / fig_h, row_in / fig_h, foot_in / fig_h
    repo = pathlib.Path(__file__).resolve().parent.parent

    for ki, (kernel, why) in enumerate(kernels):
        db_name = json.loads((repo / "bench_info" / ("%s.json" % kernel)).read_text())["benchmark"]["short_name"]
        samples = {}
        for fw, label, colour in cols:
            s = raw_samples(args.db, args.preset, db_name, fw, args.dace_variant)
            if s.size:
                samples[fw] = s
        if not samples:
            continue
        scale, unit = _unit([float(np.median(s)) for s in samples.values()])

        slot_top = 1.0 - head_f - ki * row_f
        ax_h = 0.51 * row_f
        ax_bot = slot_top - 0.85 * row_f
        row_top = ax_bot + ax_h
        for ci, (fw, label, colour) in enumerate(cols):
            ax = fig.add_axes([0.075 + ci * 0.315, ax_bot, 0.205, ax_h])
            s = samples.get(fw)
            if s is None or not s.size:
                ax.set_visible(False)
                continue
            v = s * scale
            med = float(np.median(v))

            parts = ax.violinplot([v], positions=[0], showextrema=False, showmedians=False,
                                  widths=0.9, bw_method=0.4)
            for body in parts["bodies"]:
                body.set_facecolor(colour); body.set_alpha(0.28)
                body.set_edgecolor(colour); body.set_linewidth(1.0)

            rng = np.random.default_rng(args.seed + 7 * ki + ci)
            ax.scatter(rng.uniform(-0.085, 0.085, v.size), v, s=6.0, color=colour,
                       alpha=0.55, zorder=3, linewidths=0)

            st = stats_by_pair.get((db_name, fw))
            if st:
                # The moving-block interval, not the IID one. These are sequential measurements
                # and the campaign audit found significant lag-1 autocorrelation or monotone drift
                # in 37 of 68 pairs, which is exactly the condition under which resampling
                # individual points understates the spread. Falls back to the IID interval only
                # if a stats file predating the block columns is supplied.
                lo = st.get("ci95_block_lo_s", st["ci95_median_lo_s"]) * scale
                hi = st.get("ci95_block_hi_s", st["ci95_median_hi_s"]) * scale
                # CI as a capped vertical bar OFFSET to the right of the cloud, so it is never
                # read as part of the violin
                ax.plot([0.36, 0.36], [lo, hi], color="#16232b", lw=1.6, zorder=6)
                for y in (lo, hi):
                    ax.plot([0.30, 0.42], [y, y], color="#16232b", lw=1.6, zorder=6)
            ax.plot([-0.46, 0.46], [med, med], color="#16232b", lw=1.7, zorder=7)

            ax.set_title(label, fontsize=8.6, fontweight="bold", color=colour, pad=5)
            ax.set_xticks([])
            ax.set_xlim(-0.62, 0.62)
            ax.tick_params(axis="y", labelsize=7.0)
            ax.grid(axis="y", color="#e8ebec", lw=0.5)
            ax.set_axisbelow(True)
            for sp in ("top", "right", "bottom"):
                ax.spines[sp].set_visible(False)
            if ci == 0:
                ax.set_ylabel("runtime (%s)" % unit, fontsize=8.0)
            sub = "median %.4g %s" % (med, unit)
            if st:
                sub += "\n95%% CI [%.4g, %.4g]" % (lo, hi)
                r1 = st.get("lag1_autocorr")
                if r1 is not None and abs(r1) > 2.0 / np.sqrt(v.size):
                    sub += "\nlag-1 acf %+.2f" % r1
            ax.text(0.5, -0.085, sub, transform=ax.transAxes, ha="center", va="top",
                    fontsize=6.9, color="#33424b", linespacing=1.35)

        n_s = len(next(iter(samples.values())))
        fig.text(0.075, row_top + 0.170 * row_f, "%s   -   n=%d per implementation" % (kernel, n_s),
                 fontsize=10.0, fontweight="bold", va="bottom")
        # wrapped explicitly: matplotlib's wrap=True measures against the FIGURE, not the text's
        # own anchor, so a left-anchored line runs off the right edge instead of wrapping
        fig.text(0.075, row_top + 0.088 * row_f, "\n".join(textwrap.wrap(why, width=118)),
                 fontsize=7.6, color="#455055", va="bottom", linespacing=1.3)

    fig.text(0.5, 1.0 - 0.17 / fig_h, "%s -- runtime distributions" % args.title,
             ha="center", va="top", fontsize=11.0, fontweight="bold")
    head = ("preset %s, %s. Every one of the %d samples is plotted; the violin is a kernel-density "
            "estimate over them. Thin rule = median; capped bar to its right = 95%% MOVING-BLOCK "
            "bootstrap CI of the median (%d resamples, seed %d, block 6) -- these are sequential "
            "samples, so the IID bootstrap would understate it. No outliers removed."
            % (args.preset, args.run_label, args.repeat, args.resamples, args.seed))
    for _i, _line in enumerate(textwrap.wrap(head, width=140)):
        fig.text(0.5, 1.0 - (0.40 + 0.125 * _i) / fig_h, _line, ha="center", va="top",
                 fontsize=7.2, color="#455055")
    footer = ("y-ranges differ BETWEEN implementations because their runtimes differ by up to two "
              "orders of magnitude while each distribution is tighter than 3% of its own median; a "
              "shared axis would flatten all three to lines. Units are identical within each "
              "kernel and the medians are printed, so the comparison is carried by the numbers.")
    fig.text(0.5, 0.10 / fig_h, "\n".join(textwrap.wrap(footer, width=132)),
             ha="center", va="bottom", fontsize=7.0, color="#5a666d", linespacing=1.35)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--status")
    ap.add_argument("--kernels")
    ap.add_argument("--output", required=True)
    ap.add_argument("--png-prefix", help="also write <prefix>-p1.png / -p2.png")
    ap.add_argument("--preset", default="M")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--dace-framework", default="dace_cpu_autoopt")
    ap.add_argument("--dace-variant", default=None)
    ap.add_argument("--stats", help="stats.json from slurm/stats.py; enables the distributions page")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--no-page3", action="store_true", help="write pages 1-2 only")
    ap.add_argument("--extra-violins", action="store_true",
                    help="append the EXTRA_VIOLIN_PAGES distribution pages after page 3")
    ap.add_argument("--run-label", default="",
                    help="how the run describes itself in the captions; defaults to run_kind(repeat)")
    ap.add_argument("--dace-column-label", default="DaCe\nauto_opt")
    ap.add_argument("--categories", help="JSON [[label, [kernel, ...]], ...] replacing the "
                                        "built-in PolyBench grouping")
    ap.add_argument("--title", default="NPBench PolyBench-derived kernels: Pluto vs DaCe auto_optimize")
    ap.add_argument("--subcaption", default="")
    args = ap.parse_args()

    if not args.run_label:
        args.run_label = run_kind(args.repeat)
    rows, groups, stamps = build_rows(args)
    if not args.subcaption:
        dace = sorted({v.split("+", 1)[1] for v in stamps if v and "+" in v})
        args.subcaption = ("DaCe column: %s%s   |   Pluto: polycc --tile --parallel "
                           "--codegen-context=1 (Clan frontend), clang -O3 -march=native"
                           % (args.dace_framework, (" / " + ", ".join(dace)) if dace else ""))

    cmap = diverging()
    norm = Normalize(vmin=-math.log10(RAMP), vmax=math.log10(RAMP))

    f1, agg = page_overview(rows, groups, args, cmap, norm)
    f2 = page_details(rows, args)
    dist_figs = []
    if args.stats and not args.no_page3:
        meta = json.loads(pathlib.Path(args.stats).read_text())
        args.seed = meta.get("seed", args.seed)
        args.resamples = meta.get("resamples", args.resamples)
        by_pair = {(r["db_name"], r["framework"]): r for r in meta["pairs"]}
        dist_figs.append(page_distributions(args, by_pair))
        if args.extra_violins:
            for group in EXTRA_VIOLIN_PAGES:
                dist_figs.append(page_distributions(args, by_pair, group))

    with PdfPages(args.output) as pdf:
        pdf.savefig(f1)
        pdf.savefig(f2)
        for _f in dist_figs:
            pdf.savefig(_f)
        # Carried in the document itself, not only in the printed caption: the toolchain is the
        # first thing anyone re-running these numbers needs, and a caption does not survive being
        # cropped into a thesis figure.
        info = pdf.infodict()
        info["Title"] = ("NPBench PolyBench-derived kernels at preset %s, REPEAT=%d: "
                         "Pluto vs DaCe auto_optimize" % (args.preset, args.repeat))
        info["Subject"] = args.subcaption
        info["Keywords"] = ("NPBench; PolyBench/C 4.2.1; Pluto; polycc %s (Clan frontend); "
                            "clang -O3 -march=native -fopenmp; DaCe auto_optimize; preset %s; "
                            "repeat %d" % (" ".join(POLYCC_ARGS), args.preset, args.repeat))
        info["Creator"] = "npbench/slurm/plot_thesis.py"
    if args.png_prefix:
        f1.savefig("%s-p1.png" % args.png_prefix, dpi=140)
        f2.savefig("%s-p2.png" % args.png_prefix, dpi=140)
        for _i, _f in enumerate(dist_figs):
            _f.savefig("%s-p%d.png" % (args.png_prefix, 3 + _i), dpi=190)
    plt.close(f1)
    plt.close(f2)
    for _f in dist_figs:
        plt.close(_f)
    print("wrote %s" % args.output)
    for role, g, k in agg:
        print("  geo-mean %-6s %.2fx over the %d kernels it answered" % (role, g, k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
