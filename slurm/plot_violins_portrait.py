#!/usr/bin/env python
"""Portrait 3x2 re-layout of the runtime-distribution violins from ``plot_thesis.py``.

Same measurements, same statistics, same marks -- only the arrangement differs. The landscape
page in the report puts one kernel per row and the three implementations side by side, which has
to be rotated onto a sideways thesis page. Here the grid is transposed and split two kernels at a
time: COLUMNS are kernels, ROWS are NumPy / Pluto / DaCe auto_opt, so the figure is taller than
it is wide and drops into an upright page at ``\\textwidth``.

Nothing is recomputed. Samples come from :func:`plot_thesis.raw_samples` (validated rows only),
the per-kernel unit from :func:`plot_thesis._unit`, the interval from the same
``ci95_block_*`` columns of ``stats.json``, and the jitter from the same seeding expression --
``seed + 7 * kernel_index + framework_index`` -- so every point lands exactly where it lands on
the landscape page.
"""
import argparse
import json
import pathlib
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plot_thesis
from plot_thesis import EXTRA_VIOLIN_PAGES, VIOLIN_KERNELS, raw_samples, _unit

#: Kernel -> (caption, index within its landscape page). The index is not decoration: the jitter
#: RNG is seeded with it, so reusing it is what makes the point clouds identical to the existing
#: figures rather than merely statistically equivalent.
CAPTIONS = {}
for _page in [VIOLIN_KERNELS] + EXTRA_VIOLIN_PAGES:
    for _i, (_k, _why) in enumerate(_page):
        CAPTIONS[_k] = (_why, _i)

#: Rows, top to bottom. Same colours and labels as the landscape page.
ROWS = [("numpy", "NumPy", "#b8912a"),
        ("pluto", "Pluto", "#2f7a3f"),
        ("__dace__", "DaCe auto_opt", "#2b6ca3")]

# ---- page geometry, in inches (converted to figure fractions on use) -------------------------
FIG_W = 7.2
TOP_PAD, BOT_PAD = 0.16, 0.20
TITLE_H, CAP_LINE_H, HEAD_GAP = 0.28, 0.155, 0.20
AXES_H, ANNOT_H, ROW_GAP = 1.98, 0.62, 0.18
LEFT_PAD, RIGHT_PAD = 0.44, 0.16
GUTTER, COL_TAIL = 0.74, 0.24        # y-tick room, and slack after each column's axes
BAR_X, BAR_W = 0.30, 0.055           # the colour rule that marks a row
CAP_WRAP = 44                        # characters per caption line at 8.2 pt in a 3.1 in column
CAP_LINES = 3


def build(args, kernels, stats_by_pair):
    """One 3x2 figure: ``kernels`` across, :data:`ROWS` down."""
    repo = pathlib.Path(__file__).resolve().parent.parent
    cols = [(args.dace_framework if fw == "__dace__" else fw, label, colour)
            for fw, label, colour in ROWS]

    # Everything is read before anything is drawn, so the page height can depend on the captions.
    data = []
    for kernel in kernels:
        why, ki = CAPTIONS[kernel]
        db_name = json.loads((repo / "bench_info" / ("%s.json" % kernel)).read_text())["benchmark"]["short_name"]
        samples = {}
        for fw, _label, _colour in cols:
            s = raw_samples(args.db, args.preset, db_name, fw, args.dace_variant)
            if s.size:
                samples[fw] = s
        if not samples:
            raise SystemExit("plot_violins_portrait: no validated samples for %r" % kernel)
        scale, unit = _unit([float(np.median(s)) for s in samples.values()])
        lines = textwrap.wrap(why, width=CAP_WRAP)
        if len(lines) > CAP_LINES:
            raise SystemExit("plot_violins_portrait: the caption for %r wraps to %d lines; %d fit "
                             "above the panels. Shorten it to about %d characters.\n  %s"
                             % (kernel, len(lines), CAP_LINES, CAP_WRAP * CAP_LINES, why))
        data.append(dict(kernel=kernel, db_name=db_name, ki=ki, samples=samples,
                         scale=scale, unit=unit, cap=lines))

    head_h = TITLE_H + CAP_LINE_H * CAP_LINES + HEAD_GAP
    fig_h = (TOP_PAD + head_h + len(ROWS) * (AXES_H + ANNOT_H)
             + (len(ROWS) - 1) * ROW_GAP + BOT_PAD)
    fig = plt.figure(figsize=(FIG_W, fig_h))
    x = lambda v: v / FIG_W           # inches -> figure fraction
    y = lambda v: v / fig_h           # ... measured from the BOTTOM, as matplotlib wants
    col_span = (FIG_W - LEFT_PAD - RIGHT_PAD) / len(kernels)
    axes_w = col_span - GUTTER - COL_TAIL
    head_top = fig_h - TOP_PAD

    for ri, (fw, label, colour) in enumerate(cols):
        row_top = head_top - head_h - ri * (AXES_H + ANNOT_H + ROW_GAP)
        ax_bot = row_top - AXES_H

        # A tinted band and a colour rule behind the whole row: the row, not the panel, is what
        # carries framework identity here, and it has to read as one band across both kernels.
        fig.add_artist(Rectangle((x(BAR_X), y(ax_bot - ANNOT_H + 0.06)),
                                 x(FIG_W - BAR_X - RIGHT_PAD + 0.04), y(AXES_H + ANNOT_H - 0.06),
                                 transform=fig.transFigure, facecolor=colour, alpha=0.045,
                                 edgecolor="none", zorder=0))
        fig.add_artist(Rectangle((x(BAR_X), y(ax_bot)), x(BAR_W), y(AXES_H),
                                 transform=fig.transFigure, facecolor=colour, edgecolor="none",
                                 zorder=1))
        fig.text(x(BAR_X - 0.10), y(ax_bot + AXES_H / 2.0), label, rotation=90,
                 ha="center", va="center", fontsize=11.5, fontweight="bold", color=colour)

        for kc, d in enumerate(data):
            col_x0 = LEFT_PAD + kc * col_span
            ax = fig.add_axes([x(col_x0 + GUTTER), y(ax_bot), x(axes_w), y(AXES_H)])
            ax.patch.set_alpha(0.0)
            s = d["samples"].get(fw)
            if s is None or not s.size:
                ax.set_visible(False)
                continue
            v = s * d["scale"]
            med = float(np.median(v))

            parts = ax.violinplot([v], positions=[0], showextrema=False, showmedians=False,
                                  widths=0.9, bw_method=0.4)
            for body in parts["bodies"]:
                body.set_facecolor(colour); body.set_alpha(0.28)
                body.set_edgecolor(colour); body.set_linewidth(1.0)

            # Same expression as the landscape page: kernel index within ITS page, framework index.
            rng = np.random.default_rng(args.seed + 7 * d["ki"] + ri)
            ax.scatter(rng.uniform(-0.085, 0.085, v.size), v, s=6.0, color=colour,
                       alpha=0.55, zorder=3, linewidths=0)

            st = stats_by_pair.get((d["db_name"], fw))
            if st:
                lo = st.get("ci95_block_lo_s", st["ci95_median_lo_s"]) * d["scale"]
                hi = st.get("ci95_block_hi_s", st["ci95_median_hi_s"]) * d["scale"]
                ax.plot([0.36, 0.36], [lo, hi], color="#16232b", lw=1.6, zorder=6)
                for yy in (lo, hi):
                    ax.plot([0.30, 0.42], [yy, yy], color="#16232b", lw=1.6, zorder=6)
            ax.plot([-0.46, 0.46], [med, med], color="#16232b", lw=1.7, zorder=7)

            ax.set_xticks([])
            ax.set_xlim(-0.62, 0.62)
            ax.tick_params(axis="y", labelsize=8.0)
            ax.grid(axis="y", color="#e8ebec", lw=0.5)
            ax.set_axisbelow(True)
            for sp in ("top", "right", "bottom"):
                ax.spines[sp].set_visible(False)

            sub = "median %.4g %s" % (med, d["unit"])
            if st:
                sub += "\n95%% CI [%.4g, %.4g]" % (lo, hi)
                r1 = st.get("lag1_autocorr")
                if r1 is not None and abs(r1) > 2.0 / np.sqrt(v.size):
                    sub += "\nlag-1 acf %+.2f" % r1
            ax.text(0.5, -0.075, sub, transform=ax.transAxes, ha="center", va="top",
                    fontsize=8.0, color="#33424b", linespacing=1.35)

    # Column headings last, so they sit above the row bands.
    for kc, d in enumerate(data):
        col_x0 = LEFT_PAD + kc * col_span
        n_s = len(next(iter(d["samples"].values())))
        base = head_top - TITLE_H
        fig.text(x(col_x0), y(base), "%s   (n = %d)" % (d["kernel"], n_s),
                 fontsize=12.5, fontweight="bold", color="#16232b", va="baseline")
        fig.text(x(col_x0 + col_span - COL_TAIL), y(base), "runtime (%s)" % d["unit"],
                 fontsize=8.8, color="#455055", ha="right", va="baseline")
        for li, line in enumerate(d["cap"]):
            fig.text(x(col_x0), y(base - 0.16 - CAP_LINE_H * li), line,
                     fontsize=8.2, color="#455055", va="baseline")
        rule = base - 0.16 - CAP_LINE_H * (CAP_LINES - 1) - 0.13
        fig.add_artist(Rectangle((x(col_x0), y(rule)), x(col_span - COL_TAIL), y(0.012),
                                 transform=fig.transFigure, facecolor="#c9d0d4",
                                 edgecolor="none", zorder=2))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--stats", required=True, help="stats.json from slurm/stats.py")
    ap.add_argument("--kernels", required=True, help="comma-separated bench_info stems, one per column")
    ap.add_argument("--output", required=True, help="PNG to write; must not already exist")
    ap.add_argument("--preset", default="paper")
    ap.add_argument("--dace-framework", default="dace_cpu_autoopt")
    ap.add_argument("--dace-variant", default=None)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    out = pathlib.Path(args.output)
    if out.exists():
        raise SystemExit("plot_violins_portrait: refusing to overwrite %s" % out)

    meta = json.loads(pathlib.Path(args.stats).read_text())
    args.seed = meta.get("seed", args.seed)
    by_pair = {(r["db_name"], r["framework"]): r for r in meta["pairs"]}

    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    unknown = [k for k in kernels if k not in CAPTIONS]
    if unknown:
        raise SystemExit("plot_violins_portrait: no caption on record for %s; add it to "
                         "plot_thesis.VIOLIN_KERNELS or EXTRA_VIOLIN_PAGES first"
                         % ", ".join(unknown))
    fig = build(args, kernels, by_pair)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print("wrote %s  (%.0f x %.0f px)"
          % (out, fig.get_figwidth() * args.dpi, fig.get_figheight() * args.dpi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
