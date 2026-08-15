#!/usr/bin/env python
"""Per-kernel overview for the preset-M verification campaign: NumPy baseline, Pluto and DaCe
speedups, with declined and invalid results marked rather than silently absent.

ADAPTED FROM, NOT REUSING, ``plot_results.py`` at the repo root. That script's conventions are
kept -- drop every row that did not validate, then pick each framework's BEST variant by median
time, and abbreviate speedups with an up/down arrow. Four things made it unusable directly:

  * it reads a hardcoded ``npbench.db`` from the CWD, while a campaign writes into its own
    namespace;
  * it sets ``matplotlib.rcParams['text.usetex'] = True``, and a compute node rarely has LaTeX;
  * its per-benchmark bootstrap confidence intervals are meaningless at REPEAT=1, which is what
    a verification campaign runs;
  * it has no notion of a kernel a framework DECLINED, so a Pluto kernel that was never timed is
    indistinguishable from one that was never attempted -- which is the single most important
    thing this campaign has to show.

It also cannot be imported for its helpers: the module body runs its own analysis at import time
rather than sitting behind the ``__main__`` guard.

A speedup is only ever drawn from a row whose ``validated`` flag is set. Anything else is a
labelled gap.
"""
import argparse
import json
import math
import pathlib
import sqlite3
import sys

import matplotlib

matplotlib.use("Agg")  # no display, and no LaTeX: a compute node has neither
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

#: Colours for a status cell. Deliberately not a red/green speedup ramp -- these encode whether a
#: number EXISTS, which is a different question from whether it is good.
STATUS_COLOUR = {
    "validated": "#e8f5e9",
    "declined": "#eceff1",
    "invalid": "#ffebee",
    "error": "#fff3e0",
    "missing": "#eceff1",
}


def speedup_label(x):
    """``plot_results.py``'s abbreviation: arrow + magnitude, so 0.5x reads as far from 1 as 2x."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    prefix = "↑" if x < 1 else "↓"   # up = slower than NumPy, down = faster
    v = 1.0 / x if x < 1 else x
    if v >= 1000:
        return "%s%.1fk" % (prefix, v / 1000)
    if v >= 100:
        return "%s%d" % (prefix, int(v))
    return "%s%.1f" % (prefix, v)


def runtime_label(t):
    """Seconds -> a short string, milliseconds below 0.1 s (``plot_results.py``'s rule)."""
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return ""
    return "%.2f ms" % (t * 1000) if t < 0.1 else "%.2f s" % t


def dace_build(db):
    """The DaCe build stamp recorded on the rows, e.g. ``extended@eb7b1352a``.

    ``DaceFramework.version()`` writes ``<version>+<branch>@<commit>``, so the tree a column was
    measured on is recoverable from the database itself rather than having to be asserted in a
    caption that nothing checks.
    """
    try:
        conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
        vs = [v for (v, ) in conn.execute(
            "select distinct version from results where framework = 'dace_cpu'")]
        conn.close()
    except Exception:
        return None
    stamps = sorted({v.split("+", 1)[1] for v in vs if v and "+" in v})
    return ", ".join(stamps) if stamps else None


def load(db, preset):
    """``{(kernel, framework): {variant: (median_time, validated)}}`` for one preset."""
    conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    rows = conn.execute(
        "select benchmark, framework, details, validated, time from results where preset = ?",
        (preset, )).fetchall()
    conn.close()
    acc = {}
    for bench, fw, details, validated, t in rows:
        acc.setdefault((bench, fw), {}).setdefault((details or "", bool(validated)), []).append(t)
    out = {}
    for key, variants in acc.items():
        out[key] = {k: (float(np.median(v)), k[1]) for k, v in variants.items()}
    return out


def best_valid(variants, pin=None):
    """A VALIDATED variant as ``(time, name)``; ``(None, None)`` when none qualifies.

    With ``pin`` set, ONLY that variant is eligible -- a faster sibling is ignored rather than
    substituted. That is the difference between asking "how fast is DaCe" and "how fast is THIS
    DaCe pipeline", and the thesis comparison against Pluto is the second question: Pluto is one
    fixed pipeline, so pitting it against the best of three DaCe transformations per kernel would
    compare a single optimizer against a per-kernel search.

    Without ``pin`` this is ``plot_results.py``'s rule -- fastest validated variant.
    Selecting among validated rows only is what keeps an invalid-but-fast variant from becoming
    the number a speedup is computed from.
    """
    ok = [(t, name) for (name, valid), (t, _) in variants.items() if valid and (pin is None or name == pin)]
    return min(ok) if ok else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--status", help="status.json: per kernel/framework decline reasons")
    ap.add_argument("--output", required=True)
    ap.add_argument("--preset", default="M")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--kernels", help="file with one kernel name per line (defines the row order)")
    ap.add_argument("--title", default="NPBench PolyBench-derived kernels")
    ap.add_argument("--dace-variant",
                    help="pin the DaCe SDFG variant (e.g. auto_opt). Without it, the fastest "
                         "validated variant is used, which compares Pluto against a per-kernel "
                         "search rather than against one pipeline.")
    ap.add_argument("--build", help="DaCe build label for the caption; read from the DB when omitted")
    args = ap.parse_args()

    data = load(args.db, args.preset)
    status = json.load(open(args.status)) if args.status and pathlib.Path(args.status).is_file() else {}

    if args.kernels and pathlib.Path(args.kernels).is_file():
        kernels = [l.strip() for l in open(args.kernels) if l.strip()]
    else:
        kernels = sorted({b for b, _ in data})

    # NPBench WRITES bench_info's `short_name` to the database, which is frequently not the file
    # stem: `floyd_warshall` is stored as `floydwar`, `heat_3d` as `heat3d`. Read the mapping
    # rather than guessing it -- "strip the underscores" gets heat_3d right and floyd_warshall
    # wrong, and the failure mode is a kernel silently plotted as having no result.
    repo = pathlib.Path(__file__).resolve().parent.parent

    def db_key(k):
        try:
            return json.loads((repo / "bench_info" / ("%s.json" % k)).read_text())["benchmark"]["short_name"]
        except Exception:
            return k

    rows = []
    for k in kernels:
        dk = db_key(k)
        np_t, _ = best_valid(data.get((dk, "numpy"), {}))
        entry = {"kernel": k, "numpy": np_t}
        for fw in ("pluto", "dace_cpu"):
            variants = data.get((dk, fw), {})
            pin = args.dace_variant if fw == "dace_cpu" else None
            t, variant = best_valid(variants, pin)
            st = (status.get(k, {}) or {}).get(fw, {})
            if t is not None and np_t:
                entry[fw] = {"status": "validated", "speedup": np_t / t, "time": t, "variant": variant}
            elif variants and pin and not any(n == pin and v for (n, v) in variants):
                # The kernel ran, but the PINNED variant is not among its validated results.
                # Reported as its own state: substituting a sibling here would silently answer a
                # different question than the one the pin asks.
                entry[fw] = {"status": "invalid",
                             "reason": "variant %r not validated for this kernel" % pin}
            elif variants:
                entry[fw] = {"status": "invalid", "reason": st.get("reason", "ran but did not validate")}
            else:
                entry[fw] = {
                    "status": st.get("status", "missing"),
                    "reason": st.get("reason", "no result recorded"),
                }
        rows.append(entry)

    build = args.build or dace_build(args.db)
    if args.dace_variant:
        dace_label = "DaCe CPU %s%s" % (args.dace_variant, " / %s" % build if build else "")
    else:
        dace_label = "DaCe CPU (fastest validated variant per kernel)"

    sub = ("preset %s, REPEAT=%d - VERIFICATION RUN, not a performance measurement\n"
           "DaCe column = %s" % (args.preset, args.repeat, dace_label))
    with PdfPages(args.output) as pdf:
        _bars(pdf, rows, args.title, sub, dace_label)
        _table(pdf, rows, args.title, sub)
    print("wrote %s" % args.output)


def _bars(pdf, rows, title, sub, dace_label="DaCe CPU"):
    """Speedup vs NumPy, log axis, one kernel per row. Declines are annotated, not omitted."""
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.42 * n + 2.6)))
    y = np.arange(n)
    h = 0.38
    for off, fw, colour, label in ((+h / 2, "pluto", "#1565c0", "Pluto (polycc --pet --tile --parallel)"),
                                   (-h / 2, "dace_cpu", "#ef6c00", dace_label)):
        vals, ypos = [], []
        for i, r in enumerate(rows):
            e = r[fw]
            if e["status"] == "validated":
                vals.append(e["speedup"])
                ypos.append(y[i] + off)
        ax.barh(ypos, vals, height=h, color=colour, label=label, zorder=3)
    # Everything that produced no speedup, marked in place so a gap is never ambiguous.
    for i, r in enumerate(rows):
        for off, fw in ((+h / 2, "pluto"), (-h / 2, "dace_cpu")):
            e = r[fw]
            if e["status"] != "validated":
                ax.text(1.03, y[i] + off, e["status"], va="center", ha="left", fontsize=6.5,
                        color="#b71c1c" if e["status"] == "invalid" else "#546e7a", zorder=4)
    ax.axvline(1.0, color="#37474f", lw=1.1, zorder=2)
    ax.text(1.0, -1.15, "NumPy baseline", ha="center", fontsize=7.5, color="#37474f")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([r["kernel"] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("speedup vs NumPy  (right of the line = faster than NumPy; log scale)", fontsize=9)
    ax.grid(axis="x", ls=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.set_title("%s\n%s" % (title, sub), fontsize=10.5)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _table(pdf, rows, title, sub):
    """The same data as text: NumPy runtime, both speedups, the DaCe variant, and every reason."""
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.34 * n + 2.6)))
    ax.axis("off")
    cols = ["kernel", "NumPy (baseline)", "Pluto", "DaCe CPU", "DaCe variant", "notes"]
    cells, colours = [], []
    for r in rows:
        p, d = r["pluto"], r["dace_cpu"]
        note = ""
        for fw, e in (("pluto", p), ("dace_cpu", d)):
            if e["status"] != "validated":
                note = ("%s: %s" % (fw, e.get("reason", e["status"])))[:76]
        cells.append([
            r["kernel"],
            runtime_label(r["numpy"]),
            speedup_label(p["speedup"]) if p["status"] == "validated" else p["status"],
            speedup_label(d["speedup"]) if d["status"] == "validated" else d["status"],
            d.get("variant") or "",
            note,
        ])
        colours.append([
            "white", "white",
            STATUS_COLOUR.get(p["status"], "white"),
            STATUS_COLOUR.get(d["status"], "white"), "white", "white",
        ])
    t = ax.table(cellText=cells, colLabels=cols, cellColours=colours, loc="upper center",
                 cellLoc="left", colWidths=[0.13, 0.13, 0.09, 0.09, 0.12, 0.44])
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 1.28)
    for j in range(len(cols)):
        t[0, j].set_facecolor("#cfd8dc")
        t[0, j].set_text_props(weight="bold")
    ax.set_title("%s\n%s\n↓ = faster than NumPy, ↑ = slower; only validated results carry a speedup"
                 % (title, sub), fontsize=10.5)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
