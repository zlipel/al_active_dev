"""Audit (and later correct) condensed-phase density labels by cross-checking
the EoS phase-separation call against the NPT diffusivity simulations.

Motivation
----------
The EoS caller labels a sequence non-PS (`density == 0`) when it cannot bracket
a high-density zero of P(rho). That fails silently when the pressure sweep never
reaches the repulsive branch (loop not closed / not converged). The DIFF sims
run NPT and let the box condense, so their late-time density is an independent
readout of the condensed-phase density -- reliable for genuine condensers
(replicates agree and barely fluctuate) but not for non-condensers (large
within-run and across-replicate variation, settling low).

Method (per polymer, where sim data exists)
-------------------------------------------
1. EoS reliability: pressure at the largest simulated rho, averaged over the
   later half of thermo.avg with its block std error. "Reliable" (loop closed)
   == P(rho_max) - Z*SE > 0. Negative / not-significantly-positive => suspect.
2. Diff readout: mean production density (col `density`) over the late-time
   window of each replicate log, aggregated across replicates, with within-run
   and across-replicate spreads.
3. Classify: FALSE_NEG (label non-PS but diff condensed), FALSE_POS (label PS
   but diff not condensed), plus OK / FLAG variants. Report-first -- nothing is
   written to labels until the report is reviewed.

Data availability / paths (as of this campaign)
-----------------------------------------------
- CALVADOS: no cluster sim data -> skipped here (its anomalies were corrected
  locally from the uploaded EoS sweeps).
- MPIPI: iterations 1-10 only, no seed -> seed rows (0-119) skipped.
- HPS_URRY: everything.
- Iterations absent from SCRATCH_AL are looked up under
  /projects/WEBB/from_zach/MODEL_COMPARISON/<model>/GENERATIONS/iteration_<n>.
- Seed (gen 0) sims live at $SCRATCH_AL/<model>/SIMULATIONS/{EOS,DIFF}/poly{row}.

Subcommands
-----------
report   Audit all available polymers for a model; write a per-row CSV
         (EoS reliability, diff density + spreads, classification,
         suggested_density) and print the FALSE_NEG / FALSE_POS / FLAG rows.
apply    Read an approved report CSV and write `density`/`density_std` into
         labels_gen10.csv for rows marked approved (backs up first). Run only
         after reviewing the report.

Usage (run through SLURM, not the login node -- see submit/eos_relabel.sh)
--------------------------------------------------------------------------
  sbatch submit/eos_relabel.sh report
  #   review analysis/_anomaly_step3/eos_diff_report.csv; set approved=1 on kept rows
  sbatch submit/eos_relabel.sh apply --from-report analysis/_anomaly_step3/eos_diff_report.csv
  sbatch submit/eos_relabel.sh apply --from-report analysis/_anomaly_step3/eos_diff_report.csv --apply

Direct (for local/dev only):
  python analysis/eos_relabel.py report [--model M ...] [--jobs N] [--out report.csv] \
      [--scratch DIR] [--webb DIR]
  python analysis/eos_relabel.py apply  --from-report report.csv [--apply] [--scratch DIR] [--webb DIR]
"""

import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(REPO))
from analysis.process_eos_sims import split_error  # noqa: E402

GEN0_N = 120       # seed sequences (rows 0..119)
GEN_N = 48         # sequences per AL iteration
WEBB_DEFAULT = "/projects/WEBB/from_zach/MODEL_COMPARISON"
MODELS = ["CALVADOS", "MPIPI", "HPS_URRY"]

# --- reliability / condensation thresholds (first-pass, advisory, tunable) ---
Z_SIG = 2.0        # P(rho_max) - Z_SIG*SE > 0  => EoS loop reliably closed
DIFF_TAIL_FRAC = 0.5   # fraction of each production run used for the density readout
MAG_FLOOR = 0.5    # diff density below this is treated as "not condensed"
CV_MAX = 0.15      # within-run or across-replicate CV above this => unstable


# ---------------------------------------------------------------------------
# row <-> sim-dir mapping (with SCRATCH -> WEBB fallback)
# ---------------------------------------------------------------------------
def row_scope(row):
    """(gen, local_poly). gen 0 == seed."""
    if row < GEN0_N:
        return 0, row
    return (row - GEN0_N) // GEN_N + 1, (row - GEN0_N) % GEN_N


def row_of(gen, local):
    return local if gen == 0 else GEN0_N + (gen - 1) * GEN_N + local


def gen_dir(scratch, webb, model, gen):
    """iteration_<gen> dir, preferring SCRATCH then the WEBB archive."""
    for base in (scratch, webb):
        if not base:
            continue
        d = os.path.join(base, model, "GENERATIONS", f"iteration_{gen}")
        if os.path.isdir(d):
            return d
    return None


def sim_dir(scratch, webb, model, row, kind):
    """EoS/DIFF sim dir for a global row, or None if absent. kind in {EOS,DIFF}."""
    gen, local = row_scope(row)
    if gen == 0:
        d = os.path.join(scratch, model, "SIMULATIONS", kind, f"poly{local}")
        return d if os.path.isdir(d) else None
    gd = gen_dir(scratch, webb, model, gen)
    if gd is None:
        return None
    d = os.path.join(gd, "SIMULATIONS", kind, f"poly{local}")
    return d if os.path.isdir(d) else None


def labels_file(scratch, webb, model):
    for base in (scratch, webb):
        if not base:
            continue
        p = os.path.join(base, model, "GENERATIONS", "iteration_10", "labels_gen10.csv")
        if os.path.exists(p):
            return p
    return None


def scratch_default():
    if os.environ.get("SCRATCH_AL"):
        return os.environ["SCRATCH_AL"]
    env = os.path.join(REPO, "config", "cluster.env")
    try:
        with open(env) as f:
            for line in f:
                m = re.match(r'\s*SCRATCH_AL="?([^"\n]+)"?', line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# EoS reliability (fast: only the largest-rho thermo.avg)
# ---------------------------------------------------------------------------
_THERMO_LABELS = ["TimeStep", "temp", "etot", "pe", "ke", "ent", "P", "rho"]


def eos_tail_pressure(eos_dir, frac=0.5, z=Z_SIG):
    """Pressure at the largest simulated density: mean +- block SE over the
    later `frac` of that density's thermo.avg. sig_positive == loop closed."""
    subs = []
    for name in os.listdir(eos_dir):
        m = re.match(r"rho([0-9.]+)$", name)
        if m and os.path.exists(os.path.join(eos_dir, name, "thermo.avg")):
            subs.append((float(m.group(1)), name))
    if not subs:
        return None
    _, name = max(subs)
    f = os.path.join(eos_dir, name, "thermo.avg")
    try:
        data = pd.read_csv(f, delimiter=" ", header=None, names=_THERMO_LABELS, skiprows=2).dropna()
    except Exception:
        return None
    if len(data) < 10:
        return None
    N = int((1 - frac) * len(data))
    pvals = data["P"][N:].values
    se, _ = split_error(pvals, 5)
    p_mean = float(np.mean(pvals))
    return dict(
        rho_max=float(data["rho"].values[-1]),
        P=p_mean,
        P_err=float(se),
        sig_positive=bool((p_mean - z * se) > 0),
    )


# ---------------------------------------------------------------------------
# Diff-sim density (late-time production density over replicates)
# ---------------------------------------------------------------------------
def parse_diff_log(log_path, ncol=8, dens_col=6, step_col=0):
    """Return (steps, densities) from the 8-column production thermo rows.
    thermo_style: step temp pe ke etotal press density vol -> density is col 6."""
    steps, dens = [], []
    try:
        with open(log_path) as f:
            for line in f:
                t = line.split()
                if len(t) != ncol:
                    continue
                try:
                    v = [float(x) for x in t]
                except ValueError:
                    continue
                steps.append(v[step_col])
                dens.append(v[dens_col])
    except OSError:
        return np.empty(0), np.empty(0)
    return np.asarray(steps), np.asarray(dens)


def diff_density(diff_dir, tail_frac=DIFF_TAIL_FRAC):
    """Aggregate late-time production density across all replicate runs."""
    run_means, run_stds = [], []
    for run in sorted(glob.glob(os.path.join(diff_dir, "[0-9]*"))):
        if not os.path.isdir(run):
            continue
        step, dens = parse_diff_log(os.path.join(run, "log.lammps"))
        if step.size < 10:
            continue
        cut = step.max() * (1 - tail_frac)
        m = step >= cut
        if m.sum() < 5:
            continue
        run_means.append(float(dens[m].mean()))
        run_stds.append(float(dens[m].std()))
    if not run_means:
        return None
    means = np.asarray(run_means)
    grand = float(means.mean())
    return dict(
        n_runs=len(run_means),
        density=grand,
        within_std=float(np.mean(run_stds)),
        across_std=float(means.std()),
        cv_within=float(np.mean(run_stds) / grand) if grand else np.nan,
        cv_across=float(means.std() / grand) if grand else np.nan,
        run_means=";".join(f"{x:.3f}" for x in means),
    )


def diff_is_condensed(diff):
    """Advisory: substantial magnitude + tight within-run and across-replicate
    spread. Returns True/False, or None if no diff data."""
    if diff is None:
        return None
    return bool(
        diff["density"] >= MAG_FLOOR
        and (np.isnan(diff["cv_within"]) or diff["cv_within"] <= CV_MAX)
        and (np.isnan(diff["cv_across"]) or diff["cv_across"] <= CV_MAX)
    )


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def classify(label_density, eos, diff):
    """Return (verdict, suggested_density, note)."""
    dcond = diff_is_condensed(diff)
    eos_ok = eos["sig_positive"] if eos else None  # loop closed?
    label_ps = (label_density is not None) and (label_density > 0)

    if label_density is None:
        return "NO_LABEL", np.nan, ""

    if not label_ps:  # labelled non-PS (density == 0)
        if dcond is True:
            return "FALSE_NEG", diff["density"], f"diff condensed ~{diff['density']:.2f}"
        if eos_ok is False:
            return ("FLAG_EOS_INCOMPLETE", np.nan,
                    "P(rho_max) not sig>0; diff " + ("absent" if diff is None else "inconclusive"))
        return "OK_nonPS", 0.0, ""

    # labelled PS (density > 0)
    if dcond is False:
        return "FALSE_POS", np.nan, (
            f"diff not condensed (rho~{diff['density']:.2f}, cv_w~{diff['cv_within']:.2f})"
        )
    if dcond is True and abs(diff["density"] - label_density) > max(0.2, 0.3 * label_density):
        return "FLAG_MISMATCH", diff["density"], f"diff {diff['density']:.2f} vs label {label_density:.2f}"
    return "OK_PS", label_density, ""


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def available_rows(scratch, webb, model):
    """Yield global rows that have an EoS sim dir (seed + iterations)."""
    seed = os.path.join(scratch, model, "SIMULATIONS", "EOS")
    for d in glob.glob(os.path.join(seed, "poly*")):
        m = re.search(r"poly(\d+)$", d)
        if m and int(m.group(1)) < GEN0_N:
            yield int(m.group(1))
    gens = set()
    for base in (scratch, webb):
        if not base:
            continue
        for gd in glob.glob(os.path.join(base, model, "GENERATIONS", "iteration_*")):
            gm = re.search(r"iteration_(\d+)$", gd)
            if gm and int(gm.group(1)) >= 1:
                gens.add(int(gm.group(1)))
    for g in sorted(gens):
        gd = gen_dir(scratch, webb, model, g)
        if not gd:
            continue
        for d in glob.glob(os.path.join(gd, "SIMULATIONS", "EOS", "poly*")):
            m = re.search(r"poly(\d+)$", d)
            if m:
                yield row_of(g, int(m.group(1)))


def audit_one(model, row, scratch, webb, label_density, label_exp):
    """Audit a single polymer. Module-level so joblib can pickle it."""
    gen, local = row_scope(row)
    eos = eos_tail_pressure(sim_dir(scratch, webb, model, row, "EOS") or "")
    dd = sim_dir(scratch, webb, model, row, "DIFF")
    diff = diff_density(dd) if dd else None
    verdict, suggested, note = classify(label_density, eos, diff)
    return dict(
        model=model, gen=gen, poly_local=local, row=row,
        label_density=label_density, label_exp_density=label_exp,
        eos_rho_max=round(eos["rho_max"], 3) if eos else np.nan,
        eos_P_rhomax=round(eos["P"], 3) if eos else np.nan,
        eos_P_err=round(eos["P_err"], 3) if eos else np.nan,
        eos_loop_closed=eos["sig_positive"] if eos else np.nan,
        diff_n_runs=diff["n_runs"] if diff else 0,
        diff_density=round(diff["density"], 3) if diff else np.nan,
        diff_within_std=round(diff["within_std"], 4) if diff else np.nan,
        diff_across_std=round(diff["across_std"], 4) if diff else np.nan,
        diff_run_means=diff["run_means"] if diff else "",
        verdict=verdict,
        suggested_density=round(suggested, 4) if suggested == suggested else np.nan,
        approved="",
        note=note,
    )


def cmd_report(args):
    scratch = args.scratch or scratch_default()
    webb = args.webb
    models = args.model or MODELS
    tasks = []
    for model in models:
        lf = labels_file(scratch, webb, model)
        lbl = pd.read_csv(lf) if lf else None
        if lbl is None:
            print(f"[{model}] no labels_gen10.csv found under scratch/webb — skipping", flush=True)
            continue
        found = sorted(set(available_rows(scratch, webb, model)))
        if not found:
            print(f"[{model}] no EoS sim dirs found — skipping", flush=True)
            continue
        print(f"[{model}] queuing {len(found)} polymers with EoS data ...", flush=True)
        for row in found:
            ld = float(lbl.at[row, "density"]) if row < len(lbl) else None
            le = float(lbl.at[row, "exp_density"]) if row < len(lbl) else np.nan
            tasks.append((model, row, ld, le))

    rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(audit_one)(model, row, scratch, webb, ld, le)
        for (model, row, ld, le) in tasks
    )
    df = pd.DataFrame(rows)
    out = args.out or os.path.join(REPO, "analysis", "_anomaly_step3", "eos_diff_report.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)\n")

    for model in df["model"].unique():
        sub = df[df["model"] == model]
        counts = sub["verdict"].value_counts().to_dict()
        print(f"[{model}] " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    interesting = df[df["verdict"].isin(["FALSE_NEG", "FALSE_POS", "FLAG_EOS_INCOMPLETE", "FLAG_MISMATCH"])]
    cols = ["model", "gen", "poly_local", "row", "label_density", "label_exp_density",
            "eos_P_rhomax", "eos_loop_closed", "diff_n_runs", "diff_density",
            "diff_across_std", "verdict", "suggested_density", "note"]
    print(f"\n{len(interesting)} rows need attention:")
    print(interesting[cols].to_string(index=False) if len(interesting) else "  (none)")


# ---------------------------------------------------------------------------
# apply (post-review)
# ---------------------------------------------------------------------------
def cmd_apply(args):
    scratch = args.scratch or scratch_default()
    webb = args.webb
    rep = pd.read_csv(args.from_report)

    def _is_approved(v):
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "y"):
            return True
        try:
            return float(s) > 0          # handles "1.0" from a float column
        except ValueError:
            return False

    approved = rep[rep["approved"].map(_is_approved)]
    if not len(approved):
        print("No rows marked approved (set the `approved` column to 1/yes). Nothing to do.")
        return
    print(f"{len(approved)} approved rows:")
    for model, g in approved.groupby("model"):
        lf = labels_file(scratch, webb, model)
        if not lf:
            print(f"  [{model}] no labels file found — skipping")
            continue
        df = pd.read_csv(lf)
        for _, r in g.iterrows():
            row = int(r["row"])
            new_d = float(r["suggested_density"])
            old_d = float(df.at[row, "density"])
            print(f"  {model} row {row}: density {old_d:.4f} -> {new_d:.4f}  ({r['verdict']})")
            if args.apply:
                df.at[row, "density"] = round(new_d, 6)
                if "diff_across_std" in r and r["diff_across_std"] == r["diff_across_std"]:
                    df.at[row, "density_std"] = round(float(r["diff_across_std"]), 6)
        if args.apply:
            bak = lf.replace(".csv", "_PRECORRECTION.csv")
            if not os.path.exists(bak):
                shutil.copy2(lf, bak)
                print(f"    backup -> {bak}")
            df.to_csv(lf, index=False)
            print(f"    wrote  -> {lf}")
    if not args.apply:
        print("\n(dry-run — add --apply to write)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("report", help="audit EoS vs diff; write report CSV")
    pr.add_argument("--model", action="append", help="repeatable; default all three")
    pr.add_argument("--scratch", help="SCRATCH_AL base (default: env or cluster.env)")
    pr.add_argument("--webb", default=WEBB_DEFAULT, help="archive fallback base")
    pr.add_argument("--out", help="report CSV path")
    pr.add_argument("--jobs", type=int, default=-1, help="joblib workers (default: all cores)")
    pr.set_defaults(func=cmd_report)

    pa = sub.add_parser("apply", help="write approved corrections from a report CSV")
    pa.add_argument("--from-report", required=True, help="report CSV with approved column set")
    pa.add_argument("--scratch", help="SCRATCH_AL base (default: env or cluster.env)")
    pa.add_argument("--webb", default=WEBB_DEFAULT, help="archive fallback base")
    pa.add_argument("--apply", action="store_true", help="write (default: dry-run)")
    pa.set_defaults(func=cmd_apply)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
