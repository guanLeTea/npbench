#!/bin/bash
# VERIFICATION RUN -- NPBench on CSCS Daint/Alps, native (no container).
# numpy vs pluto vs dace_cpu, preset S, over a handful of PolyBench-derived kernels.
#
# This is a SETUP CHECK, not a measurement: it exists to prove that all three columns build, run
# and VALIDATE against the NumPy reference on a compute node, with the affinity the real campaign
# will use. It is deliberately preset S and a short kernel list. Do not read speedups off it.
#
#   sbatch -A g34 ~/npbench/slurm/verify_small.sbatch
#
# Overridable: PRESET, REPEAT, TIMEOUT, KERNELS, FRAMEWORKS.
#
# ---------------------------------------------------------------------------------------------
# WHY THE CPU KNOBS ARE SPELLED TWICE
#
# --cpus-per-task is requested on the #SBATCH line AND passed explicitly to srun. Slurm 25.05 does
# NOT propagate --cpus-per-task from the batch script into the step (that implicit inheritance went
# away in 22.05). Without it on the srun line, cpus-per-task defaults to 1 and --cpu-bind=cores
# pins each rank to ONE core while the ranks still run OMP_NUM_THREADS threads -- which produces
# timings that look like measurements and are not. This was measured the hard way on the previous
# harness; it is the single most expensive mistake available here.
#
# 4 ranks x 72 CPUs = 288 = the whole node, one Grace socket per rank.
# ---------------------------------------------------------------------------------------------

#SBATCH --job-name=npbench-verify
#SBATCH --uenv=prgenv-gnu/26.3:v1   # requested from Slurm, so the mount reaches the batch step
#SBATCH --view=default              # AND every nested srun step
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # MUST equal the rank count below: srun cannot launch more
                                    # tasks than were allocated
#SBATCH --cpus-per-task=72          # see the block above -- also passed to srun explicitly
#SBATCH --exclusive                 # whole node: the four ranks split it, no foreign co-runners
#SBATCH --time=00:30:00
#SBATCH --output=results/npbench-verify/slurm-%j.out

set -uo pipefail   # NOT -e: a failing column must still reach the summary

REPO="${NPBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "${REPO}/run_benchmark.py" ]]; then
    echo "REPO=${REPO} is not an NPBench checkout (no run_benchmark.py)" >&2
    echo "  submit from the repo root, or pass NPBENCH_REPO=/path/to/npbench" >&2
    exit 2
fi
cd "${REPO}" || exit 2
# shellcheck disable=SC1090
source "${REPO}/slurm/npbench-env.sh" || exit 2
# EXPORTED, not just set: the srun body below is single-quoted and reads it from the environment.
# Without this the step dies with "REPO: unbound variable" under its `set -u`, after the ranks have
# already printed their affinity -- which looks like a successful launch in the log.
export REPO

JOB="${SLURM_JOB_ID:-local}"
export PRESET="${PRESET:-S}"
export REPEAT="${REPEAT:-5}"
export TIMEOUT="${TIMEOUT:-200}"
export FRAMEWORKS="${FRAMEWORKS:-numpy pluto dace_cpu}"

RPN="${SLURM_NTASKS_PER_NODE:-4}"; RPN="${RPN%%(*}"
export RANKS="${SLURM_NTASKS:-${RPN}}"
# CPUs per rank, and the thread count each rank gets. ONE thread per allocated CPU: the ranks
# split the node, so a rank that sized its pool from nproc (288) would oversubscribe 4x and every
# rank would fight the other three for the same cores.
CPT="${SLURM_CPUS_PER_TASK:-72}"

# One PRIVATE namespace per job. NPBench hardcodes its results database as `npbench.db` RELATIVE
# TO THE CWD (infrastructure/test.py, infrastructure/line_count.py), so four ranks sharing a
# working directory would have four processes writing one SQLite file. Each rank therefore runs
# from its OWN directory and gets its own npbench.db; they are merged at the end. Nothing else
# depends on the CWD -- run_benchmark.py puts its own directory on sys.path, and bench_info /
# framework_info / the benchmark sources are all resolved from __file__.
#
# The namespace is keyed on the JOB ID and REFUSES to reuse one that already exists. NPBench
# APPENDS to whatever npbench.db it finds, so a namespace reused across two runs silently mixes
# their rows into one database with no way to separate them afterwards.
export NS="${REPO}/results/npbench-verify/${JOB}"
if [[ -e "${NS}" ]]; then
    echo "results namespace ${NS} already exists; refusing to append into a previous run" >&2
    echo "  remove it, or submit again for a fresh job id" >&2
    exit 2
fi
mkdir -p "${NS}"

# Reproducibility stamp for the whole campaign. The DaCe branch/commit is also stamped on every
# dace_cpu ROW (infrastructure/dace_framework.py::version -> the `version` column), so a merged
# database stays interpretable even if this file is separated from it.
DACE_TREE="${DACE_TREE:-/users/levwidmer/dace}"
{
    echo "{"
    echo "  \"job\": \"${JOB}\","
    echo "  \"date\": \"$(date -Is)\","
    echo "  \"node\": \"$(hostname -s)\","
    echo "  \"preset\": \"${PRESET}\", \"repeat\": ${REPEAT},"
    echo "  \"npbench_commit\": \"$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null)\","
    echo "  \"npbench_dirty\": \"$(git -C "${REPO}" status --porcelain --untracked-files=no 2>/dev/null | wc -l)\","
    echo "  \"dace_tree\": \"${DACE_TREE}\","
    echo "  \"dace_branch\": \"$(git -C "${DACE_TREE}" rev-parse --abbrev-ref HEAD 2>/dev/null)\","
    echo "  \"dace_commit\": \"$(git -C "${DACE_TREE}" rev-parse HEAD 2>/dev/null)\","
    echo "  \"dace_dirty\": \"$(git -C "${DACE_TREE}" status --porcelain --untracked-files=no 2>/dev/null | wc -l)\","
    echo "  \"python\": \"$(python --version 2>&1)\","
    echo "  \"numpy\": \"$(python -c 'import numpy;print(numpy.__version__)' 2>/dev/null)\","
    echo "  \"dace\": \"$(python -c 'import dace;print(dace.__version__)' 2>/dev/null)\","
    echo "  \"gcc\": \"$(gcc -dumpfullversion 2>/dev/null)\","
    echo "  \"clang\": \"$(clang --version 2>/dev/null | head -1)\","
    echo "  \"polycc\": \"$(polycc --version 2>&1 | head -1)\","
    echo "  \"uenv\": \"${UENV_VIEW:-unknown}\""
    echo "}"
} > "${NS}/manifest.json"
echo "=== manifest ==="; cat "${NS}/manifest.json"

# The kernels. All PolyBench-derived and all backed by a tracked <kernel>_pluto_reference.c, so
# every one of them is a kernel all three columns can actually answer -- which is what makes a
# failure here a real failure rather than an expected gap.
# All 13 kernels the Pluto column currently answers, so the verification covers every adapter
# and every semantic correction rather than a happy-path subset:
#   heat_3d          a `constants` adapter (the scop's alpha is ABI padding)
#   jacobi_2d/seidel_2d/heat_3d   the timestep corrections in ARG_OVERRIDES
#   floyd_warshall   the int32 element-type path
cat > "${NS}/kernels.txt" <<'KERNELS'
gemm
gemver
syrk
syr2k
trmm
trisolv
mvt
lu
cholesky
jacobi_2d
seidel_2d
heat_3d
floyd_warshall
KERNELS

echo "=== npbench-verify: $(wc -l < "${NS}/kernels.txt") kernels x [${FRAMEWORKS}] preset=${PRESET} repeat=${REPEAT} ==="
echo "=== node=$(hostname -s) ranks=${RANKS} cpus-per-task=${CPT} ==="
echo "=== namespace=${NS} ==="
echo "=== dace tree=/users/levwidmer/dace branch=$(git -C /users/levwidmer/dace branch --show-current 2>/dev/null) ==="
echo "=== start $(date +%T) ==="

# DaCe writes one .so per SDFG variant per kernel. Off /tmp (tmpfs on these nodes -- the build
# would compete with the run for RAM); $SCRATCH is the parallel filesystem.
BUILD_ROOT="${DACE_BUILD_ROOT:-${SCRATCH:-${REPO}}}/npbench-verify-${JOB}"
export BUILD_ROOT

srun --ntasks="${RANKS}" --ntasks-per-node="${RPN}" --cpus-per-task="${CPT}" --cpu-bind=cores bash -c '
set -uo pipefail
rank="${SLURM_PROCID:-0}"

# One thread per CPU this rank actually owns. SLURM_CPUS_PER_TASK is what --cpus-per-task above
# reserved; nproc would report the whole node and oversubscribe it by the rank count.
threads="${SLURM_CPUS_PER_TASK:-72}"
export OMP_NUM_THREADS="${threads}"
export OPENBLAS_NUM_THREADS="${threads}"
export MKL_NUM_THREADS="${threads}"
export NUMEXPR_NUM_THREADS="${threads}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Per-rank build trees. DaCe derives an SDFG build folder from the SDFG NAME, and NPBench names
# every variant the same thing on every kernel (`fusion`, `parallel`, `auto_opt`) -- so one shared
# folder would have all four ranks overwriting one libauto_opt.so. The Pluto column has the same
# hazard with lib<kernel>_pluto.so.
export DACE_default_build_folder="${BUILD_ROOT}/rank-${rank}"
export NPBENCH_PLUTO_BUILD_DIR="${BUILD_ROOT}/pluto-rank-${rank}"
mkdir -p "${DACE_default_build_folder}" "${NPBENCH_PLUTO_BUILD_DIR}"

# Own CWD => own npbench.db (see the note above).
RUNDIR="${NS}/rank-${rank}"
mkdir -p "${RUNDIR}"
cd "${RUNDIR}" || exit 2

mapfile -t keys < "${NS}/kernels.txt"
mine=()
for i in "${!keys[@]}"; do
    [ $(( i % RANKS )) -eq "${rank}" ] && mine+=("${keys[$i]}")
done
echo "[rank ${rank}] cpus=$(taskset -cp $$ 2>/dev/null | sed "s/.*: //") threads=${threads} kernels: ${mine[*]}"

for key in "${mine[@]}"; do
    for fw in ${FRAMEWORKS}; do
        echo "[rank ${rank}] BEGIN ${fw} ${key} at $(date +%T)"
        python "${REPO}/run_benchmark.py" \
            -b "${key}" -f "${fw}" -p "${PRESET}" \
            -r "${REPEAT}" -t "${TIMEOUT}" -v True </dev/null
        echo "[rank ${rank}] END   ${fw} ${key} rc=$? at $(date +%T)"
    done
done
'
rc=$?
echo "=== srun rc=${rc} ==="

echo ""
echo "=== merging the per-rank npbench.db shards ==="
python "${REPO}/slurm/merge_db.py" --output "${NS}/npbench.db" "${NS}"/rank-*/npbench.db

echo ""
echo "=== validation summary ==="
python "${REPO}/slurm/merge_db.py" --summarize "${NS}/npbench.db"

echo ""
echo "================================================================"
echo "  DB : ${NS}/npbench.db"
echo "  LOG: ${REPO}/results/npbench-verify/slurm-${JOB}.out"
echo "================================================================"
echo "=== finish $(date +%T) ==="
exit "${rc}"
