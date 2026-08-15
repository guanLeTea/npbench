#!/usr/bin/env python
"""Classify every (kernel, framework) pair of a campaign as validated / invalid / declined / error.

The results database records only what RAN. A kernel the Pluto column declined leaves no row at
all, which is indistinguishable from a kernel that was never attempted -- so the reason is
recovered from the per-pair log the launcher captured and written next to the database.

Truth ordering, most authoritative first:

  1. a validated row in the database  -> ``validated``   (the only status that carries a speedup)
  2. a row that did not validate      -> ``invalid``
  3. a decline named in the log       -> ``declined``    (PlutoUnavailable, with its cause)
  4. anything else with a log         -> ``error``       (traceback / timeout / crash)
  5. no log at all                    -> ``missing``

The database wins over the log because a framework can print a scary line and still produce a
correct measurement; the log is only consulted when there is no row to speak for the pair.
"""
import argparse
import json
import pathlib
import re
import sqlite3

#: The framework's own decline, and its reason. Raised as PlutoUnavailable, printed either bare or
#: as the last line of a traceback.
_DECLINE = re.compile(r"PlutoUnavailable:\s*(.+)")
#: NPBench's own wording when an implementation could not be loaded or executed.
_FAILED = re.compile(r"^Failed to (?:load|execute) the (.+?) implementation\.", re.M)
_TIMEOUT = re.compile(r"timed out", re.I)


def classify_log(path: pathlib.Path):
    """``(status, reason)`` from one per-pair log."""
    if not path.is_file():
        return "missing", "no log recorded"
    text = path.read_text(errors="replace")
    m = _DECLINE.search(text)
    if m:
        reason = " ".join(m.group(1).split())
        return "declined", reason[:400]
    if _TIMEOUT.search(text):
        return "error", "timed out"
    m = _FAILED.search(text)
    if m:
        tail = [l for l in text.strip().splitlines() if l.strip()][-1:]
        return "error", (tail[0] if tail else m.group(0))[:400]
    if "Traceback" in text:
        tail = [l for l in text.strip().splitlines() if l.strip()][-1:]
        return "error", (tail[0] if tail else "traceback")[:400]
    return "error", "no result row and no recognised failure in the log"


def short_name(repo: pathlib.Path, kernel: str) -> str:
    """The name NPBench WRITES to the database for ``kernel``.

    ``bench_info/<kernel>.json`` -> ``benchmark.short_name``, which is frequently not the file
    stem: ``floyd_warshall`` is stored as ``floydwar``, ``heat_3d`` as ``heat3d``. Read rather
    than guessed -- a rule like "strip the underscores" gets ``heat_3d`` right and
    ``floyd_warshall`` wrong, and the failure mode is a kernel silently counted as missing.
    """
    p = repo / "bench_info" / ("%s.json" % kernel)
    try:
        return json.loads(p.read_text())["benchmark"]["short_name"]
    except Exception:
        return kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--preset", default="M")
    ap.add_argument("--frameworks", default="numpy pluto dace_cpu")
    ap.add_argument("--json", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    logs = pathlib.Path(args.logs)
    kernels = [l.strip() for l in open(args.kernels) if l.strip()]
    frameworks = args.frameworks.split()

    rows = []
    if pathlib.Path(args.db).is_file():
        conn = sqlite3.connect("file:%s?mode=ro" % args.db, uri=True)
        try:
            rows = conn.execute(
                "select benchmark, framework, details, validated, time from results where preset = ?",
                (args.preset, )).fetchall()
        except sqlite3.OperationalError:
            rows = []
        conn.close()

    by_pair = {}
    for bench, fw, details, validated, t in rows:
        by_pair.setdefault((bench, fw), []).append((details, bool(validated), t))

    repo = pathlib.Path(__file__).resolve().parent.parent
    status, counts = {}, {}
    for k in kernels:
        status[k] = {}
        db_name = short_name(repo, k)
        for fw in frameworks:
            got = by_pair.get((db_name, fw), [])
            if any(v for _, v, _ in got):
                st, reason = "validated", ""
            elif got:
                st, reason = "invalid", "ran but did not validate against the NumPy reference"
            else:
                st, reason = classify_log(logs / ("%s.%s.log" % (k, fw)))
            variants = sorted({d for d, _, _ in got if d})
            status[k][fw] = {"status": st, "reason": reason, "variants": variants}
            counts.setdefault(fw, {}).setdefault(st, []).append(k)

    with open(args.json, "w") as fh:
        json.dump(status, fh, indent=2, sort_keys=True)

    lines = []
    lines.append("preset %s -- %d kernels x %d frameworks" % (args.preset, len(kernels), len(frameworks)))
    lines.append("")
    lines.append("%-12s %9s %9s %9s %7s %8s" % ("framework", "validated", "invalid", "declined", "error", "missing"))
    lines.append("-" * 62)
    for fw in frameworks:
        c = counts.get(fw, {})
        lines.append("%-12s %9d %9d %9d %7d %8d" %
                     (fw, len(c.get("validated", [])), len(c.get("invalid", [])), len(c.get("declined", [])),
                      len(c.get("error", [])), len(c.get("missing", []))))
    lines.append("")
    problems = [(k, fw, v) for k, per in sorted(status.items()) for fw, v in per.items()
                if v["status"] != "validated"]
    if problems:
        lines.append("NOT VALIDATED -- every one, with its reason:")
        for k, fw, v in problems:
            lines.append("  %-16s %-10s %-9s %s" % (k, fw, v["status"], v["reason"][:150]))
    else:
        lines.append("every (kernel, framework) pair validated")
    text = "\n".join(lines) + "\n"
    with open(args.summary, "w") as fh:
        fh.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
