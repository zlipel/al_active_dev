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

## Build (Python 3.12, the project's standard)

```bash
# Activate the project conda env first (gives you pybind11, cmake, openmp):
module purge
module load anaconda3/2024.6   # or local equivalent
conda activate al_active_dev

cd external/md_calcs
mkdir -p build && cd build
cmake ..
make -j4
```

The build will produce something like `md_calcs_par.cpython-312-x86_64-linux-gnu.so`
(filename depends on the active interpreter's ABI tag) in this directory. Once it
exists, `from external.md_calcs import md_calcs_par` will resolve to it.

For other Python versions: just rebuild — the CMake configure step picks up the
active interpreter automatically via `pybind11`.

## Local development (macOS / non-cluster)

You do not need to build this on the dev box. `tests/conftest.py` registers a
shape-faithful stub at `external.md_calcs.md_calcs_par`, so the analysis tests
pass without the real extension. Tests that exercise the actual MSD math should
be marked `@pytest.mark.cluster` and run on the cluster with `CLUSTER_TESTS=1`.

If you do want a local build, the same recipe works — pybind11 + OpenMP are
included in `config/environment.yml`.
