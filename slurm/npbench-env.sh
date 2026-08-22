# NPBENCH environment for CSCS Daint/Alps (native, no container). SOURCE this, do not run it.
#
# Makes the repo-local venv, the GNU toolchain, clang/LLVM and polycc resolvable for every rank
# of an allocation, and pins the compiler DaCe hands to CMake.
#
# The uenv is requested by the JOB, not started before sbatch, so the mount reaches the batch step
# AND every nested `srun` step:
#
#     #SBATCH --uenv=prgenv-gnu/26.3:v1
#     #SBATCH --view=default
#
# Interactively (login node or a salloc shell), start the view by hand first:
#
#     uenv start --view=default prgenv-gnu/26.3:v1
#     source ~/npbench/slurm/npbench-env.sh
#
# It does NOT set PYTHONPATH: DaCe is installed into the venv as an EDITABLE install of
# ~/dace, so whichever branch that tree is on is the DaCe that runs. Check it with
# `git -C ~/dace branch --show-current` before a measurement, and stamp the answer on the run.
#
# It also does NOT set thread counts -- that is the launcher's job, because the right number
# depends on how many ranks are splitting the node (see slurm/verify_small.sbatch).
#
# Safe to source under `set -euo pipefail`: all work happens inside a function that locally
# disables errexit/nounset and restores whatever the caller had, so a failing probe cannot abort
# the job by accident. Only an EXPLICIT fatal check does, and it does so with a named cause.

# ---- site-specific paths (edit these if the layout changes) ---------------------------------
NPBENCH_ENV_REPO="${NPBENCH_ENV_REPO:-/users/levwidmer/npbench}"
NPBENCH_ENV_VENV="${NPBENCH_ENV_REPO}/.venv/bin/activate"
# LLVM 17.0.6 is the version this Pluto's pet was built against (--with-clang-prefix); it also
# supplies the clang that compiles polycc's transformed C. Keep the two in lockstep.
NPBENCH_ENV_LLVM_PREFIX="/capstor/scratch/cscs/levwidmer/opt/llvm-17.0.6"
NPBENCH_ENV_LLVM_BIN="${NPBENCH_ENV_LLVM_PREFIX}/bin"
NPBENCH_ENV_LLVM_LIB="${NPBENCH_ENV_LLVM_PREFIX}/lib"   # libomp.so + libclang-cpp.so.17 (pet)
NPBENCH_ENV_PLUTO_BIN="/capstor/scratch/cscs/levwidmer/opt/pluto/bin"   # polycc
NPBENCH_ENV_UENV_HINT="uenv start --view=default prgenv-gnu/26.3:v1"

_npbench_env_main() {
    local had_e=0 had_u=0
    case $- in *e*) had_e=1 ;; esac
    case $- in *u*) had_u=1 ;; esac
    set +eu

    local rc=0
    for _once in 1; do

        # (1) The prgenv-gnu default view must be active -- it supplies Python 3.14 (the venv's
        #     base interpreter), GCC 14.3, its libgomp, cmake and the base libraries.
        #     Tested on /user-environment/env, NOT on /user-environment: the latter is the
        #     mountpoint and exists as an empty directory even with nothing mounted, so testing
        #     it passes with no uenv and the job dies further down on the far less obvious
        #     "python is not on PATH".
        if [ ! -d /user-environment/env ]; then
            echo "npbench-env: FATAL -- the prgenv-gnu view is not active (/user-environment/env absent)." >&2
            echo "  in a BATCH job, request it from Slurm (it then reaches every srun step too):" >&2
            echo "      #SBATCH --uenv=prgenv-gnu/26.3:v1" >&2
            echo "      #SBATCH --view=default" >&2
            echo "  interactively:  ${NPBENCH_ENV_UENV_HINT}" >&2
            rc=1; break
        fi
        if command -v gcc >/dev/null 2>&1; then
            local _gccver; _gccver="$(gcc -dumpfullversion 2>/dev/null || echo '?')"
            case "${_gccver}" in
                14.*) : ;;
                *) echo "npbench-env: WARNING -- gcc is ${_gccver}, expected 14.x from prgenv-gnu/26.3." >&2 ;;
            esac
        fi

        # (2) The repo-local venv: NumPy/SciPy/pandas/matplotlib/pygount + the editable DaCe.
        if [ ! -f "${NPBENCH_ENV_VENV}" ]; then
            echo "npbench-env: FATAL -- venv activate not found: ${NPBENCH_ENV_VENV}" >&2
            echo "  create it under the uenv:  ${NPBENCH_ENV_UENV_HINT}" >&2
            echo "                             python -m venv ${NPBENCH_ENV_REPO}/.venv" >&2
            rc=1; break
        fi
        # shellcheck disable=SC1090
        source "${NPBENCH_ENV_VENV}"

        # (3) LLVM bin AFTER the venv activate, so `clang` resolves here while `python` still
        #     resolves inside the venv (the LLVM bin has no python).
        if [ -d "${NPBENCH_ENV_LLVM_BIN}" ]; then
            case ":${PATH}:" in
                *":${NPBENCH_ENV_LLVM_BIN}:"*) : ;;
                *) export PATH="${NPBENCH_ENV_LLVM_BIN}:${PATH}" ;;
            esac
        else
            echo "npbench-env: WARNING -- LLVM bin not found: ${NPBENCH_ENV_LLVM_BIN}" >&2
        fi

        # (4) polycc on PATH, and the LLVM runtime libs on LD_LIBRARY_PATH. The polycc engine
        #     (tool/pluto, pet) dynamically links clang's libomp.so and libclang-cpp.so.17 from
        #     the LLVM prefix; without this dir it fails with "libomp.so: cannot open shared
        #     object file". libgmp resolves from the prgenv-gnu view (rpath baked at build time).
        if [ -d "${NPBENCH_ENV_PLUTO_BIN}" ]; then
            case ":${PATH}:" in
                *":${NPBENCH_ENV_PLUTO_BIN}:"*) : ;;
                *) export PATH="${NPBENCH_ENV_PLUTO_BIN}:${PATH}" ;;
            esac
        else
            echo "npbench-env: WARNING -- Pluto bin not found: ${NPBENCH_ENV_PLUTO_BIN}" >&2
        fi
        if [ -d "${NPBENCH_ENV_LLVM_LIB}" ]; then
            case ":${LD_LIBRARY_PATH:-}:" in
                *":${NPBENCH_ENV_LLVM_LIB}:"*) : ;;
                *) export LD_LIBRARY_PATH="${NPBENCH_ENV_LLVM_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
            esac
        fi

        # (5) Pin the compiler DaCe hands to CMake. The prgenv-gnu view ships `gcc` and `g++`
        #     (14.3) but NOT the `cc`/`c++` aliases, so CMake's default C++ compiler search falls
        #     through the view to /usr/bin/c++ -- SUSE's GCC 7.5 -- which has no -std=c++23 and
        #     every SDFG then dies in `cmake -G Ninja`. Pinned by name so a build cannot silently
        #     pick up a different compiler than the banner below reports.
        if command -v g++ >/dev/null 2>&1; then
            export DACE_compiler_cpu_executable="$(command -v g++)"
            export DACE_compiler_linker_executable="$(command -v g++)"
        fi

        # (5b) Header and library search paths for the uenv view, so a Pluto scop that calls a
        #      numerical library finds it. `contour_integral`'s scop includes <lapacke.h>, which
        #      lives in the view's include dir; the LAPACKE entry points it calls
        #      (LAPACKE_zgetrf/zgetri/zgesv) are inside libopenblas -- this toolchain ships no
        #      separate liblapacke. CPATH and LIBRARY_PATH are the standard clang/gcc search
        #      variables, so this needs no compiler-flag change in the framework.
        local _view="/user-environment/env/default"
        if [ -d "${_view}/include" ]; then
            case ":${CPATH:-}:" in
                *":${_view}/include:"*) : ;;
                *) export CPATH="${_view}/include${CPATH:+:${CPATH}}" ;;
            esac
        fi
        if [ -d "${_view}/lib" ]; then
            case ":${LIBRARY_PATH:-}:" in
                *":${_view}/lib:"*) : ;;
                *) export LIBRARY_PATH="${_view}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}" ;;
            esac
        fi

        # (6) Everything the numpy/dace/pluto columns need must resolve, or the job stops HERE by
        #     name rather than hundreds of kernels deep in a per-kernel build error.
        local tool missing=()
        for tool in python gcc g++ cmake clang polycc; do
            command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}")
        done
        if [ "${#missing[@]}" -ne 0 ]; then
            echo "npbench-env: FATAL -- these tools are not on PATH: ${missing[*]}" >&2
            echo "  GNU tools come from the prgenv-gnu view (${NPBENCH_ENV_UENV_HINT})." >&2
            echo "  clang comes from ${NPBENCH_ENV_LLVM_BIN}; polycc from ${NPBENCH_ENV_PLUTO_BIN}." >&2
            rc=1; break
        fi

        # (7) Concise banner so the job log records exactly which toolchain measured the run.
        echo "npbench-env: ready"
        echo "  repo        ${NPBENCH_ENV_REPO}"
        echo "  python      $(python --version 2>&1)"
        echo "  numpy       $(python -c 'import numpy;print(numpy.__version__)' 2>/dev/null)"
        echo "  dace        $(python -c 'import dace;print(dace.__version__, dace.__file__)' 2>/dev/null)"
        echo "  dace branch $(git -C /users/levwidmer/dace branch --show-current 2>/dev/null)"
        echo "  gcc/g++     $(gcc -dumpfullversion 2>/dev/null) / $(g++ -dumpfullversion 2>/dev/null)"
        echo "  cmake       $(cmake --version 2>/dev/null | head -1)"
        echo "  clang       $(clang --version 2>/dev/null | head -1)"
        echo "  polycc      $(command -v polycc)"
        echo "  dace c++    ${DACE_compiler_cpu_executable:-<unpinned -- CMake will find /usr/bin/c++>}"
    done

    [ "${had_e}" = 1 ] && set -e
    [ "${had_u}" = 1 ] && set -u
    return "${rc}"
}

_npbench_env_main
_npbench_env_rc=$?
unset -f _npbench_env_main 2>/dev/null || true

if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    return "${_npbench_env_rc}" 2>/dev/null || true
else
    exit "${_npbench_env_rc}"
fi
