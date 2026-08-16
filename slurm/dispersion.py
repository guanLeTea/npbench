#!/usr/bin/env python
"""Robust dispersion per (kernel, framework) over a campaign's repetitions.

The report's speedups are MEDIANS, so the spread that matters is the one around the median, not
a standard deviation a couple of outliers can dominate. This prints the interquartile range and
the robust coefficient of variation IQR/median, which is the same statistic family the plot
already uses and needs no distributional assumption.

Only VALIDATED rows are considered. A timing from a run that did not validate is not a slow
measurement, it is not a measurement.

    python slurm/dispersion.py --db results/<ns>/npbench.db --preset L [--variant auto_opt]
"""
import argparse
import json
import pathlib
import sqlite3

import numpy as np

#: A kernel is flagged when its middle half spans more than this fraction of its median, or when
#: its extremes differ by more than this factor. Both are deliberately loose: at 20 repetitions on
#: a shared node some spread is expected, and the point is to catch rows that should not be quoted
#: to two significant figures, not to fail the campaign.
RIQR_FLAG = 0.10
RANGE_FLAG = 2.0


def short_name(repo, kernel):
    try:
        return json.loads((repo / "bench_info" / ("%s.json" % kernel)).read_text())["benchmark"]["short_name"]
    except Exception:
        return kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--preset", default="L")
    ap.add_argument("--kernels")
    ap.add_argument("--variant", help="restrict a dace framework to one SDFG variant")
    ap.add_argument("--json", help="also write the table here")
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parent.parent
    conn = sqlite3.connect("file:%s?mode=ro" % args.db, uri=True)
    rows = conn.execute(
        "select benchmark, framework, details, validated, time from results where preset = ?",
        (args.preset, )).fetchall()
    conn.close()

    acc = {}
    for bench, fw, det, valid, t in rows:
        if not valid:
            continue
        if args.variant and fw.startswith("dace") and (det or "") != args.variant:
            continue
        acc.setdefault((bench, fw), []).append(float(t))

    names = {}
    if args.kernels:
        for k in (l.strip() for l in open(args.kernels) if l.strip()):
            names[short_name(repo, k)] = k

    out, flagged = [], []
    print("%-16s %-18s %4s %12s %12s %8s %9s" %
          ("kernel", "framework", "n", "median", "IQR", "IQR/med", "max/min"))
    print("-" * 84)
    for (bench, fw), ts in sorted(acc.items()):
        a = np.asarray(ts, dtype=float)
        med = float(np.median(a))
        q1, q3 = (float(x) for x in np.percentile(a, [25, 75]))
        iqr = q3 - q1
        riqr = iqr / med if med else float("nan")
        rng = float(a.max() / a.min()) if a.min() > 0 else float("inf")
        bad = riqr > RIQR_FLAG or rng > RANGE_FLAG
        rec = {"kernel": names.get(bench, bench), "framework": fw, "n": int(a.size),
               "median_s": med, "iqr_s": iqr, "riqr": riqr, "max_over_min": rng,
               "min_s": float(a.min()), "max_s": float(a.max()), "unstable": bool(bad)}
        out.append(rec)
        print("%-16s %-18s %4d %12.6f %12.6f %7.1f%% %8.2fx%s" %
              (rec["kernel"], fw, a.size, med, iqr, 100 * riqr, rng, "  <-- unstable" if bad else ""))
        if bad:
            flagged.append(rec)

    print()
    print("%d of %d validated (kernel, framework) pairs flagged unstable "
          "(IQR/median > %.0f%% or max/min > %.1fx)" % (len(flagged), len(out), 100 * RIQR_FLAG, RANGE_FLAG))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
