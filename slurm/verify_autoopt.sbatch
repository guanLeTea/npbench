#!/bin/bash
# EQUIVALENCE CHECK -- does `dace_cpu_autoopt` reproduce `dace_cpu`'s auto_opt rows?
#
# dace_cpu_autoopt narrows DaceFramework.VARIANTS to ("auto_opt",), so it should build exactly
# the SDFG dace_cpu labels `auto_opt` and skip the other two. This runs both frameworks over the
# same kernels at preset M and compares them: same validation verdict, and timings consistent
# within run-to-run noise.
#
#   sbatch -A g34 ~/npbench/slurm/verify_autoopt.sbatch
#
# REPEAT defaults to 3 here rather than 1: the question is whether two measurements of the same
# SDFG agree, which needs some idea of the spread. This is still a correctness check, not a
# performance campaign.
#
# The kernel list is chosen to span the interesting cases rather than to be short: gemm (BLAS-
# bound), syrk/nussinov (large speedups over loop-based NumPy ports), jacobi_2d/heat_3d
# (stencils), cholesky/lu (solvers), adi (DaCe slower than NumPy), and bicg/fdtd_2d -- the two
# kernels where auto_opt was NOT the fastest variant in job 4489539, which are exactly where a
# mistaken narrowing would show up.

#SBATCH --job-name=npbench-autoopt-eq
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72          # see slurm/README.md; also passed to srun explicitly below
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=results/autoopt-eq-%j/slurm-%j.out

set -uo pipefail

REPO="${NPBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "${REPO}/run_benchmark.py" ]] || { echo "REPO=${REPO} is not an NPBench checkout" >&2; exit 2; }
cd "${REPO}" || exit 2
# shellcheck disable=SC1090
source "${REPO}/slurm/npbench-env.sh" || exit 2
export REPO

JOB="${SLURM_JOB_ID:-local}"
export PRESET="${PRESET:-M}"
export REPEAT="${REPEAT:-3}"
export TIMEOUT="${TIMEOUT:-600}"
export FRAMEWORKS="${FRAMEWORKS:-numpy dace_cpu dace_cpu_autoopt}"

RPN="${SLURM_NTASKS_PER_NODE:-4}"; RPN="${RPN%%(*}"
export RANKS="${SLURM_NTASKS:-${RPN}}"
CPT="${SLURM_CPUS_PER_TASK:-72}"

export NS="${REPO}/results/autoopt-eq-${JOB}"
mkdir -p "${NS}/logs"

cat > "${NS}/kernels.txt" <<'KERNELS'
gemm
syrk
nussinov
jacobi_2d
heat_3d
cholesky
lu
adi
bicg
fdtd_2d
KERNELS

echo "=== autoopt equivalence: $(wc -l < "${NS}/kernels.txt") kernels x [${FRAMEWORKS}] preset=${PRESET} repeat=${REPEAT}"
echo "=== node=$(hostname -s) ranks=${RANKS} cpus-per-task=${CPT}  ns=${NS}"
echo "=== npbench=$(git -C "${REPO}" rev-parse --short HEAD) dace=$(git -C /users/levwidmer/dace rev-parse --abbrev-ref HEAD)@$(git -C /users/levwidmer/dace rev-parse --short HEAD)"
echo "=== start $(date +%T)"

BUILD_ROOT="${DACE_BUILD_ROOT:-${SCRATCH:-${REPO}}}/autoopt-eq-${JOB}"
export BUILD_ROOT

srun --ntasks="${RANKS}" --ntasks-per-node="${RPN}" --cpus-per-task="${CPT}" --cpu-bind=cores bash -c '
set -uo pipefail
rank="${SLURM_PROCID:-0}"
threads="${SLURM_CPUS_PER_TASK:-72}"
export OMP_NUM_THREADS="${threads}" OPENBLAS_NUM_THREADS="${threads}"
export MKL_NUM_THREADS="${threads}" NUMEXPR_NUM_THREADS="${threads}"
export OMP_PROC_BIND=close OMP_PLACES=cores
export NPBENCH_PLUTO_BUILD_DIR="${BUILD_ROOT}/pluto-rank-${rank}"
RUNDIR="${NS}/rank-${rank}"; mkdir -p "${RUNDIR}"; cd "${RUNDIR}" || exit 2

mapfile -t keys < "${NS}/kernels.txt"
mine=(); for i in "${!keys[@]}"; do [ $(( i % RANKS )) -eq "${rank}" ] && mine+=("${keys[$i]}"); done
echo "[rank ${rank}] cpus=$(taskset -cp $$ 2>/dev/null | sed "s/.*: //") ${#mine[@]} kernels: ${mine[*]}"

for key in "${mine[@]}"; do
    for fw in ${FRAMEWORKS}; do
        # A build folder per (rank, framework): the two DaCe frameworks name their SDFG
        # `auto_opt` identically, so a shared folder would have them overwrite one
        # libauto_opt.so and the second measurement would time the first one build.
        export DACE_default_build_folder="${BUILD_ROOT}/${fw}-rank-${rank}"
        mkdir -p "${DACE_default_build_folder}"
        log="${NS}/logs/${key}.${fw}.log"
        echo "[rank ${rank}] BEGIN ${fw} ${key} $(date +%T)"
        python -u "${REPO}/run_benchmark.py" -b "${key}" -f "${fw}" -p "${PRESET}" \
               -r "${REPEAT}" -t "${TIMEOUT}" -v True </dev/null > "${log}" 2>&1
        prc=$?; echo "${prc}" > "${log%.log}.rc"
        echo "[rank ${rank}] END   ${fw} ${key} rc=${prc} $(date +%T)"
    done
done
'
rc=$?
echo "=== srun rc=${rc}"

python "${REPO}/slurm/merge_db.py" --output "${NS}/npbench.db" "${NS}"/rank-*/npbench.db

echo ""
echo "=== dace_cpu[auto_opt] vs dace_cpu_autoopt ==="
python "${REPO}/slurm/compare_autoopt.py" --db "${NS}/npbench.db" --kernels "${NS}/kernels.txt" \
    --preset "${PRESET}" | tee "${NS}/equivalence.txt"

echo ""
echo "  DIR: ${NS}"
echo "=== finish $(date +%T)"
exit "${rc}"
