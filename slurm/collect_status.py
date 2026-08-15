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
#: `.+` would stop at the first newline, which silently discards the most informative part of a
#: MULTI-LINE decline: `polycc failed on adi_pluto_reference.c:` alone says nothing, while the
#: next line -- `[Clan] Error: syntax error at line 33` -- is the actual cause. Captured to the
#: end of the log and trimmed below.
_DECLINE = re.compile(r"PlutoUnavailable:\s*(.+)", re.S)
#: NPBench's own wording when an implementation could not be loaded or executed.
_FAILED = re.compile(r"^Failed to (?:load|execute) the (.+?) implementation\.", re.M)
_TIMEOUT = re.compile(r"timed out", re.I)
#: Pluto transformed and validated, but marked no loop parallel: a correct, tiled,
#: SINGLE-THREADED row sitting beside 72-thread ones, so it has to be labelled.
_SEQUENTIAL = re.compile(r"PlutoSequential:\s*(.+)")


#: Signals worth naming when a pair's process was killed rather than exiting.
_SIGNALS = {4: "SIGILL", 6: "SIGABRT", 8: "SIGFPE", 9: "SIGKILL", 11: "SIGSEGV", 15: "SIGTERM"}


def classify_log(path: pathlib.Path, rc=None):
    """``(status, reason)`` from one per-pair log and, when known, its exit code.

    The exit code is consulted FIRST for a signal death. A framework whose generated binary
    crashes takes the interpreter with it, and a process killed by a signal never flushes its
    buffered output -- so the log of the pair that most needs explaining is the one most likely
    to be empty. Without the code, that is indistinguishable from a run that printed nothing.
    """
    if rc is not None and rc >= 128:
        sig = rc - 128
        return "error", ("process killed by signal %d (%s) -- the framework's generated binary "
                         "crashed at runtime; no output survived" % (sig, _SIGNALS.get(sig, "unknown")))
    if not path.is_file():
        return "missing", "no log recorded"
    text = path.read_text(errors="replace")
    m = _DECLINE.search(text)
    if m:
        # NPBench keeps printing after the traceback ("Failed to run ... validation."); the
        # decline message is everything up to that, flattened onto one line.
        body = m.group(1).split("\nFailed to ")[0]
        reason = " ".join(body.split())
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


def sequential_note(path: pathlib.Path) -> str:
    """The framework's own ``PlutoSequential`` line for this pair, or ``""``.

    Read even when the pair VALIDATED, unlike everything else here: it is not a failure, it is a
    qualifier on a good measurement. Pluto tiled the kernel but found no parallelism, so the
    number is single-threaded and must not be read as a 72-thread speedup.
    """
    if not path.is_file():
        return ""
    m = _SEQUENTIAL.search(path.read_text(errors="replace"))
    return " ".join(m.group(1).split())[:300] if m else ""


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
    ap.add_argument("--joblog", help="slurm job log; exit codes are recovered from its "
                                     "`END <fw> <kernel> rc=<n>` lines when no .rc sidecar exists")
    ap.add_argument("--dace-variant",
                    help="count dace_cpu as validated only when THIS SDFG variant validated "
                         "(e.g. auto_opt). Without it any validated variant counts.")
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

    # Exit codes: the .rc sidecar the launcher writes, falling back to the job log for a campaign
    # that predates it.
    rcs = {}
    if args.joblog and pathlib.Path(args.joblog).is_file():
        for m in re.finditer(r"END\s+(\S+)\s+(\S+)\s+rc=(\d+)", pathlib.Path(args.joblog).read_text(errors="replace")):
            rcs[(m.group(2), m.group(1))] = int(m.group(3))

    repo = pathlib.Path(__file__).resolve().parent.parent
    status, counts = {}, {}
    for k in kernels:
        status[k] = {}
        db_name = short_name(repo, k)
        for fw in frameworks:
            got = by_pair.get((db_name, fw), [])
            pin = args.dace_variant if fw.startswith("dace") else None
            # With a pin, a sibling variant validating is NOT an answer to the question asked:
            # the campaign is comparing one named pipeline, not the best of several.
            hits = [(d, v, t) for d, v, t in got if pin is None or d == pin]
            if any(v for _, v, _ in hits):
                st, reason = "validated", ""
            elif got and pin and not hits:
                st, reason = "invalid", "variant %r not present for this kernel" % pin
            elif got:
                st, reason = "invalid", "ran but did not validate against the NumPy reference"
            else:
                rc_file = logs / ("%s.%s.rc" % (k, fw))
                rc = None
                if rc_file.is_file():
                    try:
                        rc = int(rc_file.read_text().strip())
                    except ValueError:
                        rc = None
                if rc is None:
                    rc = rcs.get((k, fw))
                st, reason = classify_log(logs / ("%s.%s.log" % (k, fw)), rc)
            variants = sorted({d for d, _, _ in got if d})
            note = sequential_note(logs / ("%s.%s.log" % (k, fw)))
            status[k][fw] = {"status": st, "reason": reason, "variants": variants,
                             "sequential": bool(note), "note": note}
            counts.setdefault(fw, {}).setdefault(st, []).append(k)

    with open(args.json, "w") as fh:
        json.dump(status, fh, indent=2, sort_keys=True)

    lines = []
    lines.append("preset %s -- %d kernels x %d frameworks" % (args.preset, len(kernels), len(frameworks)))
    if args.dace_variant:
        lines.append("dace_cpu pinned to the %r SDFG variant (siblings ignored, not substituted)"
                     % args.dace_variant)
    lines.append("")
    lines.append("%-12s %9s %9s %9s %7s %8s" % ("framework", "validated", "invalid", "declined", "error", "missing"))
    lines.append("-" * 62)
    for fw in frameworks:
        c = counts.get(fw, {})
        lines.append("%-12s %9d %9d %9d %7d %8d" %
                     (fw, len(c.get("validated", [])), len(c.get("invalid", [])), len(c.get("declined", [])),
                      len(c.get("error", [])), len(c.get("missing", []))))
    lines.append("")
    seq = [(k, fw) for k, per in sorted(status.items()) for fw, v in per.items()
           if v.get("sequential")]
    if seq:
        lines.append("VALIDATED BUT SINGLE-THREADED (Pluto found no parallelism; tiled only):")
        for k, fw in seq:
            lines.append("  %-16s %s" % (k, fw))
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
