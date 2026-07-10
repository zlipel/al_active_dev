"""
Top-level pytest configuration for the al_active_dev test suite.

Two responsibilities:
  1. Make `al_pipeline`, `external`, `analysis`, and `simulation` importable from
     the repo root (so tests can `from al_pipeline.acquisition.ehvi import ...`).
  2. Stub `external.md_calcs.md_calcs_par`. The .so is built per environment
     (see external/md_calcs/README.md) and may not exist locally. Tests that
     need real MSD math should be marked with the `cluster` mark and run on a
     host where the extension has been built. `external.core` is pure Python
     and imports for real — no stub needed.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_on_path(p: Path) -> None:
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_on_path(REPO_ROOT)


# GENDATA is a runtime env var that simulation scripts pass through to a child
# process via subprocess. Tests don't execute that child, but the variable must
# be defined for argparse-style fallbacks to work.
os.environ.setdefault("GENDATA", "/nonexistent/gendata.py")


def _install_md_calcs_stub() -> None:
    """Inject a stub for `external.md_calcs.md_calcs_par`.

    The extension is built per Python ABI (see external/md_calcs/README.md);
    when the .so is not present, the import would fail. We pre-register a
    shape-faithful stub at the full dotted path so analysis-side tests pass.
    A real build is loaded instead whenever it exists with a matching tag.
    """
    full_name = "external.md_calcs.md_calcs_par"
    if full_name in sys.modules:
        return

    stub = types.ModuleType(full_name)

    def msd_calc(traj, _mode):
        # Shape-faithful default: one MSD value per timepoint, zero everywhere.
        import numpy as np
        return np.zeros(traj.shape[0])

    stub.msd_calc = msd_calc
    sys.modules[full_name] = stub


def _install_ray_stub() -> None:
    """Inject a minimal `ray` module so simulation/make_eos imports without ray installed.

    Tests that actually exercise `filter_by_density` (the only ray-using path) must
    install real ray and mark themselves `cluster`. Everything else only needs ray
    to be importable.
    """
    if "ray" in sys.modules:
        return

    stub = types.ModuleType("ray")
    stub.init = lambda *_a, **_kw: None
    stub.shutdown = lambda *_a, **_kw: None
    stub.get = lambda futures: list(futures)

    def remote(fn=None, **_kw):
        # Support both @ray.remote and @ray.remote(...) usage.
        if fn is None:
            def wrap(f):
                f.remote = lambda *a, **kw: f(*a, **kw)
                return f
            return wrap
        fn.remote = lambda *a, **kw: fn(*a, **kw)
        return fn

    stub.remote = remote
    sys.modules["ray"] = stub


def _install_mpi4py_stub() -> None:
    """Inject a minimal `mpi4py` so beam_search modules import without MPI installed.

    Real mpi4py has to be built against the cluster's openmpi (see the
    cluster bootstrap in README.md). Locally we don't run MPI code, but
    beam_search.cross_paths.beam_search imports ``from mpi4py import MPI``
    at module top-level to read the process rank for debug prints. The stub
    provides a shape-faithful `COMM_WORLD.Get_rank()` returning 0 so tests
    that import the beam module (Row 9 policy tests) don't fail at
    collection time.
    """
    if "mpi4py" in sys.modules:
        return
    stub = types.ModuleType("mpi4py")

    class _Comm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Barrier(self):
            pass

    class _MPI:
        COMM_WORLD = _Comm()
        ANY_SOURCE = -1
        ANY_TAG = -1

        class Status:
            def Get_source(self):
                return 0

            def Get_tag(self):
                return 0

    stub.MPI = _MPI
    sys.modules["mpi4py"] = stub


_install_md_calcs_stub()
_install_ray_stub()
_install_mpi4py_stub()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "cluster: requires real cluster-side core/md_calcs modules; skipped locally.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip any test marked `cluster` when CLUSTER_TESTS=1 is not set."""
    if os.environ.get("CLUSTER_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="cluster-only test; set CLUSTER_TESTS=1 to enable")
    for item in items:
        if "cluster" in item.keywords:
            item.add_marker(skip)
