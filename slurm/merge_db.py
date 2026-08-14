#!/usr/bin/env python
"""Merge per-rank NPBench result databases, and summarize one.

NPBench writes its results to ``npbench.db`` RELATIVE TO THE CWD
(``infrastructure/test.py``, ``infrastructure/line_count.py``). A multi-rank job therefore gives
each rank its own working directory -- four processes writing one SQLite file is a corruption
hazard, not merely a contention one -- which leaves a shard per rank to merge afterwards.

    python slurm/merge_db.py --output merged.db results/.../rank-*/npbench.db
    python slurm/merge_db.py --summarize merged.db

The merge drops each shard's ``id`` and lets the destination reassign it: the ranks number their
rows from 1 independently, so keeping them would collide on the primary key. It is idempotent per
invocation -- the output is rebuilt from scratch -- so merging twice cannot double the rows.
"""
import argparse
import os
import pathlib
import sqlite3
import sys

TABLES = ("results", "lcounts")


def merge(shards, output):
    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Rebuilt from scratch, so a re-run cannot append the same rows twice.
    if out.exists():
        out.unlink()

    dst = sqlite3.connect(str(out))
    created = set()
    merged = 0
    for shard in shards:
        p = pathlib.Path(shard)
        if not p.is_file():
            print("skip (missing): {}".format(p), file=sys.stderr)
            continue
        src = sqlite3.connect("file:{}?mode=ro".format(p), uri=True)
        for table in TABLES:
            row = src.execute("select sql from sqlite_master where type='table' and name=?", (table, )).fetchone()
            if row is None:
                continue
            if table not in created:
                dst.execute(row[0])
                created.add(table)
            cols = [c[1] for c in src.execute("pragma table_info({})".format(table))]
            keep = [c for c in cols if c != "id"]
            sel = "select {} from {}".format(", ".join(keep), table)
            ins = "insert into {} ({}) values ({})".format(table, ", ".join(keep), ", ".join("?" * len(keep)))
            rows = src.execute(sel).fetchall()
            dst.executemany(ins, rows)
            merged += len(rows)
        src.close()
    dst.commit()
    dst.close()
    print("merged {} rows from {} shard(s) into {}".format(merged, len(shards), out))


def summarize(db):
    p = pathlib.Path(db)
    if not p.is_file():
        print("no database at {}".format(p), file=sys.stderr)
        return 1
    conn = sqlite3.connect("file:{}?mode=ro".format(p), uri=True)
    # A job whose ranks all died before writing leaves an EMPTY database, not a missing one. Say so
    # plainly rather than raising OperationalError out of the summary step.
    if conn.execute("select name from sqlite_master where type='table' and name='results'").fetchone() is None:
        print("no `results` table in {} -- every rank failed before writing a row".format(p), file=sys.stderr)
        return 1
    rows = conn.execute("""
        select benchmark, framework, details,
               min(validated) as ok, count(*) as n, median_time
        from (
            select benchmark, framework, details, validated, time,
                   avg(time) over (partition by benchmark, framework, details) as median_time
            from results
        )
        group by benchmark, framework, details
        order by benchmark, framework, details
    """).fetchall()
    if not rows:
        print("no result rows")
        return 1
    print("{:<16} {:<10} {:<10} {:>6} {:>5} {:>12}".format("benchmark", "framework", "variant", "valid", "n",
                                                           "mean (ms)"))
    failures = 0
    for benchmark, framework, details, ok, n, mean_t in rows:
        valid = "yes" if ok else "NO"
        if not ok:
            failures += 1
        print("{:<16} {:<10} {:<10} {:>6} {:>5} {:>12.3f}".format(benchmark, framework, details or "", valid, n,
                                                                  (mean_t or 0.0) * 1000.0))
    print("")
    if failures:
        print("VALIDATION FAILURES: {} (benchmark, framework, variant) combinations".format(failures))
        return 1
    print("all {} (benchmark, framework, variant) combinations validated".format(len(rows)))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="*", help="per-rank npbench.db files to merge")
    ap.add_argument("--output", help="destination database for the merge")
    ap.add_argument("--summarize", metavar="DB", help="print a validation/timing summary of DB")
    args = ap.parse_args()

    rc = 0
    if args.summarize:
        rc = summarize(args.summarize)
    elif args.output:
        merge(args.shards, args.output)
    else:
        ap.error("pass either --output (with shards) or --summarize DB")
    sys.exit(rc)
