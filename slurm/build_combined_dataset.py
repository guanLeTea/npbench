#!/usr/bin/env python
"""Assemble a DERIVED dataset: NumPy and Pluto rows from one campaign, DaCe rows from another.

Used once, to produce ``results/nonpoly-final50-dace-fixed``. Four DaCe kernels of the
31-kernel campaign (``vadv``, ``cavity_flow``, ``nbody``, ``covariance2``) were fixed after
that campaign ran. Patching four rows into it would have left the DaCe column split across two
DaCe revisions, so the DaCe column was re-measured in full at the newer revision and swapped in
whole; NumPy and Pluto were reused unchanged.

    python slurm/build_combined_dataset.py <base_dir> <dace_dir> <out_dir>

``base_dir`` supplies NumPy and Pluto, ``dace_dir`` supplies every DaCe row. Neither input is
modified and the script refuses to write into an existing ``out_dir``. Provenance -- both job
ids, both DaCe revisions, both NPBench revisions -- is written to ``manifest.json`` and
``README.md`` in the output, because the result is not reproducible from any single Slurm job.

Statistics and figures are regenerated from the output directory afterwards with stats.py,
dispersion.py and plot_thesis.py, exactly as for a real campaign.
"""
import datetime
import json
import pathlib
import shutil
import sqlite3
import sys

FW = "dace_cpu_autoopt"


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    base, dace, out = (pathlib.Path(p) for p in sys.argv[1:4])
    if out.exists():
        sys.exit("refusing to overwrite %s" % out)
    out.mkdir(parents=True)

    # The database: every base row except DaCe, plus the DaCe rows of the second campaign.
    shutil.copy2(base / "npbench.db", out / "npbench.db")
    con = sqlite3.connect(str(out / "npbench.db"))
    dropped = con.execute("select count(*) from results where framework=?", (FW, )).fetchone()[0]
    con.execute("delete from results where framework=?", (FW, ))
    cols = [c[1] for c in con.execute("pragma table_info(results)") if c[1] != "id"]
    src = sqlite3.connect("file:%s?mode=ro" % (dace / "npbench.db"), uri=True)
    rows = src.execute("select %s from results where framework=?" % ",".join(cols), (FW, )).fetchall()
    con.executemany("insert into results (%s) values (%s)" % (",".join(cols), ",".join("?" * len(cols))), rows)
    con.commit()
    print("DaCe rows: %d removed, %d inserted" % (dropped, len(rows)))
    print("combined:", dict(con.execute("select framework, count(*) from results group by framework")))
    con.close()
    src.close()

    # The status file the plots read, merged the same way.
    status = json.loads((base / "status.json").read_text())
    new = json.loads((dace / "status.json").read_text())
    for kernel, entry in status.items():
        if kernel in new and FW in new[kernel]:
            entry[FW] = new[kernel][FW]
    (out / "status.json").write_text(json.dumps(status, indent=1, sort_keys=True))
    shutil.copy2(base / "kernels.txt", out / "kernels.txt")

    mb = json.loads((base / "manifest.json").read_text())
    md = json.loads((dace / "manifest.json").read_text())
    (out / "manifest.json").write_text(json.dumps({
        "derived": True,
        "what": "NumPy and Pluto rows from the base campaign; every DaCe row re-measured at a "
                "newer DaCe revision. This is a derived comparison dataset, not a single Slurm "
                "campaign.",
        "built": datetime.datetime.now().astimezone().isoformat(),
        "numpy_pluto_source_job": mb["job"], "numpy_pluto_source_dir": str(base),
        "dace_source_job": md["job"], "dace_source_dir": str(dace),
        "dace_revision_old": mb["dace_commit"], "dace_revision_new": md["dace_commit"],
        "npbench_revision_base": mb["npbench_commit"], "npbench_revision_dace": md["npbench_commit"],
        "preset": mb["preset"], "repeat": mb["repeat"], "kernels": mb["kernels"],
        "frameworks": "numpy pluto dace_cpu_autoopt",
        "ranks": mb["ranks"], "cpus_per_task": mb["cpus_per_task"],
        "dace_rows_replaced": "all %s (complete DaCe re-measurement, not a per-kernel patch)" % mb["kernels"],
    }, indent=1))
    (out / "README.md").write_text(
        "# Derived dataset: %s\n\n"
        "**Not a single Slurm campaign.** Assembled from two runs so that the DaCe column is\n"
        "internally consistent at one DaCe revision.\n\n"
        "| rows | source job | NPBench | DaCe |\n|---|---|---|---|\n"
        "| NumPy, Pluto | %s | `%s` | `%s` |\n| DaCe auto_opt (all %s) | %s | `%s` | `%s` |\n\n"
        "Preset %s, REPEAT=%s, %s kernels, %s ranks x %s CPUs.\n\n"
        "Rebuild with `slurm/build_combined_dataset.py <base> <dace> <out>`, then regenerate\n"
        "statistics and figures with stats.py, dispersion.py and plot_thesis.py.\n"
        % (out.name, mb["job"], mb["npbench_commit"][:9], mb["dace_commit"][:9], mb["kernels"],
           md["job"], md["npbench_commit"][:9], md["dace_commit"][:9],
           mb["preset"], mb["repeat"], mb["kernels"], mb["ranks"], mb["cpus_per_task"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
