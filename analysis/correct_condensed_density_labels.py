"""Correct the stale condensed-phase density labels (the `density` column) in
the final cumulative label file `labels_gen10.csv`, for the anomaly rows that
Step 3 of the anomaly analysis diagnosed as false negatives.

Only `density` / `density_std` are touched. `exp_density` (the expenditure
density = obj1) is left as-is per the user: most expenditure densities are fine
and the optimization objective is barely affected.

Scope: the cumulative `labels_gen10.csv` per model (NOT the intermediate
labels_gen1..9). Rows are addressed by global CSV row index.

The corrected value is the condensed-phase density (largest root of P(rho)=0)
recomputed by re-running the CURRENT EoS caller on the uploaded trajectory in
`runs/<MODEL>/anomalies/poly<row>/`. We use the drill-down twin of
`bootstrap_eos_analysis`, which was validated identical to the unmodified
pipeline caller (CALVADOS poly68: both return rho_star=1.379).

Two anomalies are NOT corrected here because their uploaded trajectories are
grid-truncated (P<0 at every simulated density -> no repulsive branch captured,
so no condensed density can be bracketed). They are listed in PENDING_SIMS and
require the EoS grid to be extended to higher density first.

Usage:
    python analysis/correct_condensed_density_labels.py            # dry-run (print table)
    python analysis/correct_condensed_density_labels.py --apply    # write, with backup
"""

import os
import shutil
import sys

import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(REPO))
from analysis.process_eos_sims import get_EOS  # noqa: E402
from analysis.anomaly_pofrho_drilldown import drilldown  # noqa: E402

# (model, global_row_idx) -> trajectory at runs/<model>/anomalies/poly<row>.
# Branch captured (n_pos > 0) -> correctable from data on hand.
CORRECTIONS = [
    ("CALVADOS", 68),
    ("CALVADOS", 83),
    ("CALVADOS", 118),
    ("MPIPI", 68),
    ("MPIPI", 277),
    ("MPIPI", 331),
]

# Grid-truncated uploads: no positive branch on the simulated grid. Correct
# these only after the EoS sims are extended to higher density and re-uploaded.
PENDING_SIMS = [
    ("MPIPI", 174),
    ("HPS_URRY", 68),
]

NBOOT = 1000
SEED = 12345


def labels_path(model):
    return os.path.join(REPO, "runs", model, "GENERATIONS", "iteration_10", "labels_gen10.csv")


def recompute_density(model, row):
    """Return (density, density_std, n_pos, exp_new) from the uploaded traj."""
    traj = os.path.join(REPO, "runs", model, "anomalies", f"poly{row}")
    P, err, rho = get_EOS(traj, frac=0.5, bootstrap=True)
    res = drilldown(rho, P, nboot=NBOOT, seed=SEED)
    n_pos = int(np.sum(res["P_mean"] > 0))
    return res["rho_star_mean"], res["rho_star_std"], n_pos, res["exp_density_mean"]


def main(apply):
    # Group corrections by model so each file is read/written once.
    by_model = {}
    for model, row in CORRECTIONS:
        by_model.setdefault(model, []).append(row)

    print(f"{'model':9s} {'row':>4s} {'old_density':>12s} {'new_density':>12s} "
          f"{'new_std':>8s} {'n_pos':>5s} {'exp(unchanged)':>14s}")
    print("-" * 72)

    for model, rows in by_model.items():
        path = labels_path(model)
        df = pd.read_csv(path)
        for row in rows:
            new_d, new_s, n_pos, exp_new = recompute_density(model, row)
            old_d = df.at[row, "density"]
            exp_old = df.at[row, "exp_density"]
            flag = "" if n_pos > 0 else "  <-- WARNING: no positive branch, skipping"
            print(f"{model:9s} {row:4d} {old_d:12.4f} {new_d:12.4f} {new_s:8.4f} "
                  f"{n_pos:5d} {exp_old:8.3f}(->{exp_new:.3f}){flag}")
            if apply and n_pos > 0:
                df.at[row, "density"] = round(float(new_d), 6)
                df.at[row, "density_std"] = round(float(new_s), 6)
        if apply:
            bak = path.replace(".csv", "_PRECORRECTION.csv")
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
                print(f"  backup -> {bak}")
            df.to_csv(path, index=False)
            print(f"  wrote  -> {path}")

    print("\nPENDING (grid-truncated uploads; need extended EoS sims first):")
    for model, row in PENDING_SIMS:
        path = labels_path(model)
        df = pd.read_csv(path)
        print(f"  {model:9s} row {row:4d}  campaign density={df.at[row,'density']:.4f}  "
              f"exp_density={df.at[row,'exp_density']:.4f}")

    if not apply:
        print("\n(dry-run — re-run with --apply to write labels_gen10.csv)")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
