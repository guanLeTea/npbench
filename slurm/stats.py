#!/usr/bin/env python
"""Per (kernel, framework) runtime statistics for a campaign, with bootstrap CIs for the MEDIAN.

Reports n, median, IQR, min/max and a 95% bootstrap confidence interval of the median, plus flags
for distributions that should not be quoted without a second look. Timing samples are right-skewed
and frequently multimodal, so nothing here assumes normality: the CI is a percentile bootstrap of
the median rather than mean +/- t*sd.

VALIDATED rows only. A timing from a run that did not validate is not a slow measurement, it is
not a measurement, and it never enters a statistic here.

Reproducible: the resampling uses a fixed seed (--seed, default 20260817) and a fixed number of
resamples (--resamples, default 10000), both recorded in the JSON so a rerun can be checked
against it.

    python slurm/stats.py --db results/<ns>/npbench.db --preset paper \
        --kernels results/<ns>/kernels.txt --variant auto_opt \
        --json results/<ns>/stats.json --csv results/<ns>/stats.csv
"""
import argparse
import csv
import json
import pathlib
import sqlite3

import numpy as np

#: Flag thresholds. Deliberately loose -- these mark rows to look at, not rows to drop. NOTHING
#: here removes an outlier: a bimodal kernel is a fact about the measurement, and trimming it
#: would hide exactly what the flag exists to surface.
RIQR_FLAG = 0.10        # middle half spans >10% of the median
RANGE_FLAG = 2.0        # slowest sample >2x the fastest
CI_FLAG = 0.05          # 95% CI of the median wider than 5% of the median
OUTLIER_FLAG = 3.0      # samples beyond median +/- 3 x IQR
GAP_FLAG = 0.25         # largest gap in sorted samples > 25% of the range -> likely multimodal
#: ...but only once the sample actually has spread worth splitting. On a distribution whose middle
#: half spans 0.1% of its median, a single stray point owns most of the range and the gap statistic
#: fires on essentially every pair, which is noise rather than a finding. Multimodality is only
#: reported when the range is at least this wide relative to the median.
GAP_MIN_SPREAD = 0.02


def bootstrap_median_ci(samples, resamples, seed, alpha=0.05):
    """Percentile bootstrap CI for the median. Returns (lo, hi)."""
    rng = np.random.default_rng(seed)
    n = samples.size
    if n < 2:
        return float(samples[0]), float(samples[0])
    idx = rng.integers(0, n, size=(resamples, n))
    meds = np.median(samples[idx], axis=1)
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def lag1_autocorr(samples):
    """Lag-1 autocorrelation in EXECUTION order. |r| > 2/sqrt(n) is significant at 95%."""
    x = samples - samples.mean()
    d = float(np.sum(x * x))
    return float(np.sum(x[1:] * x[:-1]) / d) if d > 0 else 0.0


def drift_spearman(samples):
    """Rank correlation of runtime against repetition index: monotone drift over the run."""
    n = samples.size
    i = np.arange(n)
    ri = np.argsort(np.argsort(i)).astype(float)
    rv = np.argsort(np.argsort(samples)).astype(float)
    ri -= ri.mean(); rv -= rv.mean()
    d = float(np.sqrt(np.sum(ri * ri) * np.sum(rv * rv)))
    return float(np.sum(ri * rv) / d) if d > 0 else 0.0


def block_bootstrap_median_ci(samples, resamples, seed, block=6, alpha=0.05):
    """Moving-block bootstrap CI for the median.

    The IID bootstrap resamples individual points and so assumes the 50 timings are
    exchangeable. They are not: these are sequential measurements on a shared node, and the
    Paper x 50 campaign shows lag-1 autocorrelation up to +0.66 and monotone drift up to 2.2%
    across a run (thermal and contention effects, not noise). Resampling contiguous BLOCKS keeps
    the local correlation structure inside each block, which is what makes the interval honest
    when neighbouring samples are related.

    ``block`` is the block length in samples; 6 is roughly where the autocorrelation of these
    series has decayed.
    """
    rng = np.random.default_rng(seed)
    n = samples.size
    L = max(2, min(block, n))
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(resamples, nb))
    meds = np.empty(resamples)
    for i in range(resamples):
        meds[i] = np.median(np.concatenate([samples[s:s + L] for s in starts[i]])[:n])
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), L


def multimodality_gap(samples):
    """Largest gap between consecutive sorted samples, as a fraction of the full range.

    A crude but honest multimodality probe: a unimodal sample of 50 has its points fairly evenly
    spread, while two clusters leave one large empty interval between them. It flags for a human,
    it does not classify.
    """
    s = np.sort(samples)
    rng = float(s[-1] - s[0])
    if rng <= 0 or s.size < 3:
        return 0.0
    return float(np.max(np.diff(s)) / rng)


def short_name(repo, kernel):
    try:
        return json.loads((repo / "bench_info" / ("%s.json" % kernel)).read_text())["benchmark"]["short_name"]
    except Exception:
        return kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--preset", default="paper")
    ap.add_argument("--kernels")
    ap.add_argument("--variant", help="restrict a dace framework to one SDFG variant")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--block", type=int, default=6, help="moving-block bootstrap block length")
    ap.add_argument("--json")
    ap.add_argument("--csv")
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parent.parent
    conn = sqlite3.connect("file:%s?mode=ro" % args.db, uri=True)
    rows = conn.execute(
        "select benchmark, framework, details, validated, time from results where preset = ?",
        (args.preset, )).fetchall()
    conn.close()

    acc = {}
    dropped = 0
    for bench, fw, det, valid, t in rows:
        if not valid:
            dropped += 1
            continue
        if args.variant and fw.startswith("dace") and (det or "") != args.variant:
            dropped += 1
            continue
        acc.setdefault((bench, fw), []).append(float(t))

    names = {}
    if args.kernels:
        for k in (l.strip() for l in open(args.kernels) if l.strip()):
            names[short_name(repo, k)] = k

    out = []
    for (bench, fw), ts in sorted(acc.items()):
        a = np.asarray(ts, dtype=float)
        med = float(np.median(a))
        q1, q3 = (float(x) for x in np.percentile(a, [25, 75]))
        iqr = q3 - q1
        lo, hi = bootstrap_median_ci(a, args.resamples, args.seed)
        b_lo, b_hi, blk = block_bootstrap_median_ci(a, args.resamples, args.seed, args.block)
        r1 = lag1_autocorr(a)
        rho = drift_spearman(a)
        gap = multimodality_gap(a)
        n_out = int(np.sum((a < med - OUTLIER_FLAG * iqr) | (a > med + OUTLIER_FLAG * iqr))) if iqr > 0 else 0
        riqr = iqr / med if med else float("nan")
        rng = float(a.max() / a.min()) if a.min() > 0 else float("inf")
        ci_w = (hi - lo) / med if med else float("nan")
        flags = []
        if riqr > RIQR_FLAG: flags.append("wide-IQR")
        if rng > RANGE_FLAG: flags.append("wide-range")
        if ci_w > CI_FLAG: flags.append("wide-CI")
        spread = (float(a.max() - a.min()) / med) if med else 0.0
        if gap > GAP_FLAG and spread > GAP_MIN_SPREAD: flags.append("possible-multimodal")
        if n_out: flags.append("outliers=%d" % n_out)
        # serial structure: an IID bootstrap is not justified when either fires
        if abs(r1) > 2.0 / np.sqrt(a.size): flags.append("autocorrelated(r1=%+.2f)" % r1)
        if abs(rho) > 0.35: flags.append("drift(rho=%+.2f)" % rho)
        out.append({
            "kernel": names.get(bench, bench), "db_name": bench, "framework": fw,
            "n": int(a.size), "median_s": med, "q1_s": q1, "q3_s": q3, "iqr_s": iqr,
            "min_s": float(a.min()), "max_s": float(a.max()),
            "ci95_median_lo_s": lo, "ci95_median_hi_s": hi,
            "ci95_block_lo_s": b_lo, "ci95_block_hi_s": b_hi, "block_len": blk,
            "lag1_autocorr": r1, "drift_spearman": rho,
            "iqr_over_median": riqr, "max_over_min": rng, "ci95_width_over_median": ci_w,
            "largest_gap_frac": gap, "range_over_median": spread,
            "n_outliers_3iqr": n_out, "flags": flags,
        })

    hdr = "%-16s %-18s %4s %12s %12s %8s %26s %s"
    print(hdr % ("kernel", "framework", "n", "median", "IQR", "IQR/med", "95% CI of median", "flags"))
    print("-" * 122)
    for r in out:
        print(hdr % (r["kernel"], r["framework"], r["n"], "%.6f" % r["median_s"], "%.6f" % r["iqr_s"],
                     "%.1f%%" % (100 * r["iqr_over_median"]),
                     "[%.6f, %.6f]" % (r["ci95_median_lo_s"], r["ci95_median_hi_s"]),
                     ",".join(r["flags"])))
    flagged = [r for r in out if r["flags"]]
    print()
    print("%d of %d validated pairs carry a flag; %d non-validated/other-variant rows excluded"
          % (len(flagged), len(out), dropped))
    print("bootstrap: %d resamples, seed %d, percentile method, no outlier removal"
          % (args.resamples, args.seed))

    meta = {"preset": args.preset, "seed": args.seed, "resamples": args.resamples,
            "method": "percentile bootstrap of the median, 95%",
            "method_serial": "moving-block bootstrap of the median, 95%% CI, block=%d" % args.block,
            "note": ("these are sequential measurements; where lag1_autocorr or drift_spearman is "
                     "flagged the IID percentile interval understates the uncertainty and the "
                     "block interval should be quoted instead"),
            "outlier_removal": "none",
            "validated_rows_only": True, "excluded_rows": dropped, "pairs": out}
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(meta, indent=2, sort_keys=False))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            cols = [c for c in out[0] if c != "flags"] + ["flags"]
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in out:
                row = dict(r); row["flags"] = ";".join(r["flags"])
                w.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
