#!/usr/bin/env python
"""Compare ``dace_cpu``'s ``auto_opt`` rows against ``dace_cpu_autoopt``'s.

``dace_cpu_autoopt`` narrows ``DaceFramework.VARIANTS`` to ``("auto_opt",)``, so it should build
the same SDFG ``dace_cpu`` labels ``auto_opt`` and merely skip the other two. Two things have to
hold for that to be true, and this checks both:

  * the VALIDATION verdict must agree -- the same SDFG cannot be correct under one name and wrong
    under the other;
  * the timings must agree within run-to-run noise. They cannot be expected to match exactly:
    each is a separate compile and a separate set of samples on a shared node.

A timing gap is reported but is NOT by itself a failure -- it is flagged for a human to read
against the observed spread. A validation disagreement is a hard failure, because it would mean
the narrowing changed what was built.
"""
import argparse
import json
import pathlib
import sqlite3
import statistics
import sys


def rows_for(conn, preset):
    out = {}
    for bench, fw, det, valid, t in conn.execute(
            "select benchmark, framework, details, validated, time from results where preset = ?",
        (preset, )):
        out.setdefault((bench, fw, det), []).append((bool(valid), t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--preset", default="M")
    ap.add_argument("--tolerance", type=float, default=25.0,
                    help="percent gap in median time above which a row is flagged for reading")
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parent.parent
    kernels = [l.strip() for l in open(args.kernels) if l.strip()]

    def short(k):
        try:
            return json.loads((repo / "bench_info" / ("%s.json" % k)).read_text())["benchmark"]["short_name"]
        except Exception:
            return k

    conn = sqlite3.connect("file:%s?mode=ro" % args.db, uri=True)
    data = rows_for(conn, args.preset)
    conn.close()

    print("%-14s %12s %12s %9s %8s %8s  %s" %
          ("kernel", "dace_cpu", "autoopt", "gap", "valid_a", "valid_b", "verdict"))
    print("-" * 88)
    hard, flagged, compared = [], [], 0
    for k in kernels:
        b = short(k)
        a = data.get((b, "dace_cpu", "auto_opt"))
        c = data.get((b, "dace_cpu_autoopt", "auto_opt"))
        if not a or not c:
            print("%-14s %12s %12s %9s %8s %8s  %s" %
                  (k, "-" if not a else "ok", "-" if not c else "ok", "", "", "",
                   "MISSING rows (a=%s b=%s)" % (bool(a), bool(c))))
            hard.append((k, "missing rows"))
            continue
        va, vb = any(v for v, _ in a), any(v for v, _ in c)
        ta, tb = statistics.median([t for _, t in a]), statistics.median([t for _, t in c])
        gap = (tb - ta) / ta * 100.0 if ta else float("nan")
        compared += 1
        verdict = "ok"
        if va != vb:
            verdict = "VALIDATION MISMATCH"
            hard.append((k, "validation %s vs %s" % (va, vb)))
        elif abs(gap) > args.tolerance:
            verdict = "timing gap > %.0f%%" % args.tolerance
            flagged.append((k, gap))
        print("%-14s %12.3f %12.3f %8.1f%% %8s %8s  %s" %
              (k, ta * 1000, tb * 1000, gap, va, vb, verdict))

    print()
    print("compared %d kernels" % compared)
    if hard:
        print("HARD FAILURES (the narrowing changed what was built):")
        for k, why in hard:
            print("   %-14s %s" % (k, why))
    else:
        print("validation verdicts agree on every compared kernel")
    if flagged:
        print("timing gaps beyond %.0f%% -- read against the observed spread, not automatically a failure:"
              % args.tolerance)
        for k, g in flagged:
            print("   %-14s %+.1f%%" % (k, g))
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
