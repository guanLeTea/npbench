#!/usr/bin/env python
"""Two-page thesis report for a PolyBench-derived NPBench campaign.

Page 1 -- a compact speedup heatmap in the style of the HPCAgent-Bench figure: one row per
kernel, columns ``Pluto | DaCe auto_opt | NumPy``, kernels grouped by PolyBench category. Colour
is a diverging RdYlGn ramp on ``log10(speedup)`` centred at 1x, so "faster than NumPy" trends
green, "slower" trends red and 1x is neutral. The log is what makes 0.5x read as far from centre
as 2x; a linear ramp would squeeze every regression into a narrow band next to neutral.

Page 2 -- the diagnostics behind page 1: per kernel, the NumPy runtime, both speedups or their
status, the validation verdict, and the reason a cell is blank. Page 1 is the comparison; page 2
is why some of it is missing.

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

    known = [k for _, ks in CATEGORY for k in ks]
    if args.kernels and pathlib.Path(args.kernels).is_file():
        wanted = [l.strip() for l in open(args.kernels) if l.strip()]
    else:
        wanted = known
    groups = []
    for label, ks in CATEGORY:
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
        ax.text(-0.12, y + 0.5, r["kernel"], ha="right", va="center", fontsize=8.0)

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
             % (args.preset, args.repeat, run_kind(args.repeat)),
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

    xs = [0.0, 0.095, 0.220, 0.305, 0.420, 0.487]
    heads = ["kernel", "NumPy runtime", "Pluto", "DaCe auto_opt", "validated", "note / reason a cell is blank"]
    for x, h in zip(xs, heads):
        ax.text(x, 0.55, h, fontsize=7.8, fontweight="bold", va="center")
    ax.plot([0, 1], [0.95, 0.95], color="#5b666d", lw=1.0)

    for i, r in enumerate(rows):
        y = i + 1.35
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, y - 0.48), 1, 0.96, facecolor="#f4f6f7", edgecolor="none", zorder=0))
        p, d = r["pluto"], r["dace"]
        ax.text(xs[0], y, r["kernel"], fontsize=7.8, va="center", zorder=2)
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

        note = ""
        for role, e in (("Pluto", p), ("DaCe", d)):
            if e["status"] != "validated":
                note = "%s: %s" % (role, short_reason(e["status"], e.get("reason", "")))
                break
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
    fig.text(0.5, 0.958,
             "preset %s, REPEAT=%d, %s. Explains every blank cell on page 1; "
             "a speedup is shown only where the result validated."
             % (args.preset, args.repeat, run_kind(args.repeat)),
             ha="center", va="top", fontsize=7.6, color="#455055")
    for _i, _line in enumerate(caption_lines(args.subcaption)):
        fig.text(0.5, 0.937 - 0.0115 * _i, _line, ha="center", va="top",
                 fontsize=7.0, color="#455055")
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
    ap.add_argument("--dace-column-label", default="DaCe\nauto_opt")
    ap.add_argument("--title", default="NPBench PolyBench-derived kernels: Pluto vs DaCe auto_optimize")
    ap.add_argument("--subcaption", default="")
    args = ap.parse_args()

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
    with PdfPages(args.output) as pdf:
        pdf.savefig(f1)
        pdf.savefig(f2)
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
    plt.close(f1)
    plt.close(f2)
    print("wrote %s" % args.output)
    for role, g, k in agg:
        print("  geo-mean %-6s %.2fx over the %d kernels it answered" % (role, g, k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
