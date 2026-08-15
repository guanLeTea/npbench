#!/bin/bash
# PRESET-M VERIFICATION CAMPAIGN (M x 1) -- the 23 PolyBench-derived NPBench kernels,
# NumPy vs Pluto vs DaCe CPU (~/dace @ extended), on CSCS Daint/Alps.
#
# This is a VERIFICATION run, not the performance campaign: REPEAT=1, so every timing is a
# single sample and no ranking should be read off it. What it establishes is that all three
# columns build, run and VALIDATE at preset M, and what each Pluto decline actually is.
#
#   sbatch -A g34 ~/npbench/slurm/m_verify.sbatch
#
# Overridable: PRESET, REPEAT, TIMEOUT, FRAMEWORKS, KERNELS_FILE.
#
# A Pluto-declined kernel is RECORDED, never forced: the framework raises PlutoUnavailable
# with a named cause, this script captures that cause per (kernel, framework) into status.json,
# and the plot marks the row instead of leaving an ambiguous gap. Forcing a timing there would
# mean timing something polycc did not produce.
#
# See slurm/README.md for the uenv, the per-rank hazards and the affinity rules; the
# --cpus-per-task duplication below is load-bearing and explained there.

#SBATCH --job-name=npbench-m-verify
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # MUST equal the rank count used below
#SBATCH --cpus-per-task=72          # 4 x 72 = 288 = whole node, one Grace socket per rank.
                                    # Slurm 25.05 does not propagate this into the step, so it
                                    # is ALSO passed to srun explicitly further down.
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=results/m-verify-%j/slurm-%j.out

set -uo pipefail   # NOT -e: a failing column must still reach the summary and the plot

REPO="${NPBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "${REPO}/run_benchmark.py" ]] || { echo "REPO=${REPO} is not an NPBench checkout" >&2; exit 2; }
cd "${REPO}" || exit 2
# shellcheck disable=SC1090
source "${REPO}/slurm/npbench-env.sh" || exit 2
export REPO   # the srun body is single-quoted and reads this from the environment

JOB="${SLURM_JOB_ID:-local}"
export PRESET="${PRESET:-M}"
export REPEAT="${REPEAT:-1}"
export TIMEOUT="${TIMEOUT:-600}"
# The thesis comparison: NumPy baseline, Pluto, and DaCe CPU auto_optimize as ONE named
# pipeline. dace_cpu_autoopt builds only the auto_opt SDFG, so this campaign is
# structurally incapable of reporting the best of three DaCe transformations.
export FRAMEWORKS="${FRAMEWORKS:-numpy pluto dace_cpu_autoopt}"

RPN="${SLURM_NTASKS_PER_NODE:-4}"; RPN="${RPN%%(*}"
export RANKS="${SLURM_NTASKS:-${RPN}}"
CPT="${SLURM_CPUS_PER_TASK:-72}"

# The campaign namespace. Keyed on the job id and REFUSED if it already exists: NPBench APPENDS
# to whatever npbench.db it finds, so a reused namespace silently mixes two campaigns.
export NS="${REPO}/results/m-verify-${JOB}"
if [[ -e "${NS}" && -n "$(ls -A "${NS}" 2>/dev/null | grep -v '^slurm-' || true)" ]]; then
    echo "campaign namespace ${NS} already has artifacts; refusing to append into it" >&2
    exit 2
fi
mkdir -p "${NS}/logs"

# The 23 PolyBench-derived kernels, i.e. exactly those carrying a tracked
# <kernel>_pluto_reference.c. Derived from the tree rather than typed out, so this cannot drift
# away from what the Pluto column can actually be asked about.
KERNELS_FILE="${KERNELS_FILE:-${NS}/kernels.txt}"
if [[ ! -s "${KERNELS_FILE}" ]]; then
    find "${REPO}/npbench/benchmarks/polybench" -name '*_pluto_reference.c' -printf '%f\n' \
        | sed 's/_pluto_reference\.c$//' | sort > "${KERNELS_FILE}"
fi
NK=$(wc -l < "${KERNELS_FILE}")

DACE_TREE="${DACE_TREE:-/users/levwidmer/dace}"
{
    echo "{"
    echo "  \"campaign\": \"m-verify\", \"job\": \"${JOB}\","
    echo "  \"purpose\": \"preset-M correctness verification at REPEAT=1; not a performance measurement\","
    echo "  \"date\": \"$(date -Is)\", \"node\": \"$(hostname -s)\","
    echo "  \"preset\": \"${PRESET}\", \"repeat\": ${REPEAT}, \"timeout_s\": ${TIMEOUT},"
    echo "  \"kernels\": ${NK}, \"frameworks\": \"${FRAMEWORKS}\","
    echo "  \"ranks\": ${RANKS}, \"cpus_per_task\": ${CPT},"
    echo "  \"npbench_commit\": \"$(git -C "${REPO}" rev-parse HEAD 2>/dev/null)\","
    echo "  \"npbench_branch\": \"$(git -C "${REPO}" rev-parse --abbrev-ref HEAD 2>/dev/null)\","
    echo "  \"npbench_dirty_files\": $(git -C "${REPO}" status --porcelain --untracked-files=no 2>/dev/null | wc -l),"
    echo "  \"dace_tree\": \"${DACE_TREE}\","
    echo "  \"dace_branch\": \"$(git -C "${DACE_TREE}" rev-parse --abbrev-ref HEAD 2>/dev/null)\","
    echo "  \"dace_commit\": \"$(git -C "${DACE_TREE}" rev-parse HEAD 2>/dev/null)\","
    echo "  \"dace_dirty_files\": $(git -C "${DACE_TREE}" status --porcelain --untracked-files=no 2>/dev/null | wc -l),"
    echo "  \"python\": \"$(python --version 2>&1)\","
    echo "  \"numpy\": \"$(python -c 'import numpy;print(numpy.__version__)' 2>/dev/null)\","
    echo "  \"dace\": \"$(python -c 'import dace;print(dace.__version__)' 2>/dev/null)\","
    echo "  \"gcc\": \"$(gcc -dumpfullversion 2>/dev/null)\","
    echo "  \"clang\": \"$(clang --version 2>/dev/null | head -1)\","
    echo "  \"polycc\": \"$(polycc --version 2>&1 | head -1)\","
    echo "  \"uenv\": \"${UENV_VIEW:-unknown}\""
    echo "}"
} > "${NS}/manifest.json"

echo "=== npbench m-verify: ${NK} kernels x [${FRAMEWORKS}] preset=${PRESET} repeat=${REPEAT} ==="
echo "=== namespace=${NS}"
echo "=== node=$(hostname -s) ranks=${RANKS} cpus-per-task=${CPT}"
echo "=== npbench=$(git -C "${REPO}" rev-parse --short HEAD) dace=$(git -C "${DACE_TREE}" rev-parse --abbrev-ref HEAD)@$(git -C "${DACE_TREE}" rev-parse --short HEAD)"
echo "=== start $(date +%T)"
cat "${NS}/manifest.json"

BUILD_ROOT="${DACE_BUILD_ROOT:-${SCRATCH:-${REPO}}}/m-verify-${JOB}"
export BUILD_ROOT

srun --ntasks="${RANKS}" --ntasks-per-node="${RPN}" --cpus-per-task="${CPT}" --cpu-bind=cores bash -c '
set -uo pipefail
rank="${SLURM_PROCID:-0}"
threads="${SLURM_CPUS_PER_TASK:-72}"
export OMP_NUM_THREADS="${threads}" OPENBLAS_NUM_THREADS="${threads}"
export MKL_NUM_THREADS="${threads}" NUMEXPR_NUM_THREADS="${threads}"
export OMP_PROC_BIND=close OMP_PLACES=cores

# Per-rank build trees: DaCe names every SDFG variant identically on every kernel, and the
# Pluto column names every library lib<kernel>_pluto.so, so a shared folder has the ranks
# overwriting each other.
export DACE_default_build_folder="${BUILD_ROOT}/dace-rank-${rank}"
export NPBENCH_PLUTO_BUILD_DIR="${BUILD_ROOT}/pluto-rank-${rank}"
mkdir -p "${DACE_default_build_folder}" "${NPBENCH_PLUTO_BUILD_DIR}"

# Own CWD => own npbench.db, merged at the end.
RUNDIR="${NS}/rank-${rank}"; mkdir -p "${RUNDIR}"; cd "${RUNDIR}" || exit 2

mapfile -t keys < "${NS}/kernels.txt"
mine=(); for i in "${!keys[@]}"; do [ $(( i % RANKS )) -eq "${rank}" ] && mine+=("${keys[$i]}"); done
echo "[rank ${rank}] cpus=$(taskset -cp $$ 2>/dev/null | sed "s/.*: //") threads=${threads} ${#mine[@]} kernels: ${mine[*]}"

for key in "${mine[@]}"; do
    for fw in ${FRAMEWORKS}; do
        log="${NS}/logs/${key}.${fw}.log"
        echo "[rank ${rank}] BEGIN ${fw} ${key} $(date +%T)"
        # Full per-pair log: this is where a Pluto decline reason is preserved. Nothing is
        # retried or forced -- a decline is the result.
        #
        # `python -u`: a Pluto-transformed kernel can SEGFAULT the interpreter (measured on
        # nussinov, rc=139). Block-buffered stdout is lost when a process dies by signal, so
        # without -u the log of the one pair that most needs explaining arrives empty.
        python -u "${REPO}/run_benchmark.py" -b "${key}" -f "${fw}" -p "${PRESET}" \
               -r "${REPEAT}" -t "${TIMEOUT}" -v True </dev/null > "${log}" 2>&1
        prc=$?
        # The exit code beside the log: a signal death (128+N) is otherwise indistinguishable
        # from a clean run that happened to print nothing.
        echo "${prc}" > "${log%.log}.rc"
        echo "[rank ${rank}] END   ${fw} ${key} rc=${prc} $(date +%T)"
    done
done
'
rc=$?
echo "=== srun rc=${rc} ==="

echo ""
echo "=== merging per-rank databases ==="
python "${REPO}/slurm/merge_db.py" --output "${NS}/npbench.db" "${NS}"/rank-*/npbench.db

echo ""
echo "=== classifying every (kernel, framework) pair ==="
python "${REPO}/slurm/collect_status.py" \
    --logs "${NS}/logs" --kernels "${NS}/kernels.txt" --db "${NS}/npbench.db" \
    --preset "${PRESET}" --frameworks "${FRAMEWORKS}" \
    --joblog "${NS}/slurm-${JOB}.out" \
    --json "${NS}/status.json" --summary "${NS}/summary.txt"
cat "${NS}/summary.txt"

echo ""
echo "=== plot ==="
python "${REPO}/slurm/plot_m_verify.py" --db "${NS}/npbench.db" --status "${NS}/status.json" \
    --kernels "${NS}/kernels.txt" --preset "${PRESET}" --repeat "${REPEAT}" \
    --dace-framework "${DACE_FRAMEWORK:-dace_cpu_autoopt}" \
    --output "${NS}/m-verify-${JOB}.pdf"

echo ""
echo "================================================================"
echo "  DIR      : ${NS}"
echo "  DB       : ${NS}/npbench.db"
echo "  MANIFEST : ${NS}/manifest.json"
echo "  STATUS   : ${NS}/status.json"
echo "  SUMMARY  : ${NS}/summary.txt"
echo "  PLOT     : ${NS}/m-verify-${JOB}.pdf"
echo "  LOG      : ${NS}/slurm-${JOB}.out"
echo "================================================================"
echo "=== finish $(date +%T)"
exit "${rc}"
