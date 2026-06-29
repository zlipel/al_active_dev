# collect_results_model.py

import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch_dir", required=True)
    parser.add_argument("--model",       required=True, choices=["HPS_URRY", "MPIPI", "CALVADOS"])
    parser.add_argument("--length_changes", action='store_true', help="Whether length-changing edits were allowed")
    args = parser.parse_args()

    base_dir = os.path.join(args.scratch_dir, "PATHS", args.model, "RESULTS") if args.length_changes else os.path.join(args.scratch_dir, "PATHS_FIXED_LENGTH", args.model, "RESULTS")

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
    out_csv = os.path.join(args.scratch_dir, "PATHS", args.model, f"paths_master_{args.model}.csv") if args.length_changes else os.path.join(args.scratch_dir, "PATHS_FIXED_LENGTH", args.model, f"paths_master_{args.model}.csv")
    master.to_csv(out_csv, index=False)
    print(f"Wrote {len(master)} rows to {out_csv}")

if __name__ == "__main__":
    main()
