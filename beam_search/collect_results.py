# collect_results_model.py

import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch_dir", required=True)
    parser.add_argument("--model",       required=True, choices=["HPS_URRY", "MPIPI", "CALVADOS"])
    parser.add_argument("--mode", choices=["benchmark", "production"], default="production",
                        help="Which run's results to collect. Mirrors run_beams_mpi.py's "
                             "<MODE> layer under the PATHS tree.")
    parser.add_argument("--policy",
                        choices=["expert_tied", "anchored_reject", "soft", "hard", "global"],
                        default="expert_tied",
                        help="Policy subfolder written by the runner under <MODE>/.")
    parser.add_argument("--length_changes", action='store_true', help="Whether length-changing edits were allowed")
    args = parser.parse_args()

    # Mirror run_beams_mpi.py's paths_dir: <scratch>/<LENGTH_DIR>/<MODEL>/<MODE>/<POLICY>/.
    # RESULTS/ and paths_master live inside that policy folder
    length_dir = "PATHS" if args.length_changes else "PATHS_FIXED_LENGTH"
    policy_dir = os.path.join(args.scratch_dir, length_dir, args.model, args.mode.upper(), args.policy)
    base_dir = os.path.join(policy_dir, "RESULTS")

    all_rows = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == "paths.csv":
                p = os.path.join(root, f)
                df = pd.read_csv(p)
                df["source_file"] = os.path.relpath(p, base_dir)
                all_rows.append(df)

    if not all_rows:
        print(f"No paths.csv files found under {base_dir}")
        return

    master = pd.concat(all_rows, ignore_index=True)
    out_csv = os.path.join(policy_dir, f"paths_master_{args.model}.csv")
    master.to_csv(out_csv, index=False)
    print(f"Wrote {len(master)} rows to {out_csv}")

if __name__ == "__main__":
    main()
