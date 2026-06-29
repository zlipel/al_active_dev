# external/md_calcs — MSD/VACF C++ extension (vendored)

This directory holds the pybind11 C++ extension used by
`analysis/process_diff_sims.py` to compute MSDs (and a few related quantities).
It is vendored from `/home/zl4808/scripts/md_analysis/` on Stellar so the
project is self-contained.

## Contents

| File | Purpose |
|---|---|
| `md_calcs_parallel.cpp` | pybind11 source. Exports `msd_calc`, `msd_xdir_calc`, `msd_ydir_calc`, `msd_zdir_calc`, `vacf_calc`, `shear_stress_calc`, `compute_moduli`, `vacf_calc_notime`, `vacf_calc_decade`, `msd_calc_decade`, `sacf_total`. |
| `CMakeLists.txt` | Build recipe (pybind11 + OpenMP). |
| `__init__.py` | Makes this directory a Python subpackage of `external.md_calcs`. |

The compiled `.so` is **not** vendored — it depends on the Python ABI and is
expected to be built per-environment. The build drops it here, beside the source.

## Build on the cluster

The build needs four things from the cluster's module system: a Python (via
anaconda3), a C++ compiler (intel), OpenMP (comes with the compiler), and CMake.
Conda's own `cxx-compiler` / `openmp` / `cmake` are deliberately **not** in
`config/environment.yml` — using them would link against conda's libstdc++ and
clash with the cluster's runtime libraries.

```bash
# load everything in this order
module purge
module load anaconda3/2024.6
module load intel/2022.2
module load cmake
conda activate al_active_dev

# build
cd external/md_calcs
mkdir -p build && cd build
cmake ..
make -j4
ls ../md_calcs_par*.so   # e.g. md_calcs_par.cpython-312-x86_64-linux-gnu.so
```

The CMake configure step auto-discovers the active Python via `pybind11`, so the
.so naturally targets whichever Python you're in. Re-run from `build/` whenever
the Python version changes.

## Local development (macOS / non-cluster)

You do not need to build this on the dev box. `tests/conftest.py` registers a
shape-faithful stub at `external.md_calcs.md_calcs_par`, so the analysis tests
pass without the real extension. Tests that exercise the actual MSD math should
be marked `@pytest.mark.cluster` and run on a host where the .so exists, with
`CLUSTER_TESTS=1`.

If you do want a local build, install `cmake` and `pybind11` via pip/brew/conda
and follow the same recipe. The compiler/OpenMP step depends on your platform
(Apple Clang's OpenMP support is fiddly; gcc/g++ from homebrew is simpler).
