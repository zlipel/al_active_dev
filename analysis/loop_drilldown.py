"""P(rho) drill-down for the report's flagged sequences.

Same machinery as the Step-3 anomaly drill-down, but driven by the
eos_diff_report.csv flags instead of a hardcoded list. For every flagged row
whose EoS sweep has been copied to runs/<model>/anomalies/poly<globalrow>/, it
re-runs the bootstrap EoS analysis and records the diagnostics the report's
one-point loop test cannot see:

  - sep_frac : fraction of bootstrap P(rho) curves that clear the significance
               test -- the caller's actual PS decision (>= 0.5 => PS).
  - rho_star : bootstrapped condensed-phase density from the EoS.
  - exp_density : bootstrapped expenditure density.

and overlays them on a P(rho) panel per sequence (with the bootstrap envelope),
so a shallow, marginally-"significant" dip is visually and statistically
distinguishable from a real van der Waals loop.

Outputs under analysis/_anomaly_step3/:
  loop_drilldown_<MODEL>.png   per-model panel grid
  loop_drilldown_summary.csv   sep_frac / rho_star / exp vs report + label

Usage:
  python analysis/loop_drilldown.py [--report FILE] [--verdicts FALSE_NEG,...] [--nboot N]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(REPO))
from analysis.process_eos_sims import get_EOS  # noqa: E402
from analysis.anomaly_pofrho_drilldown import drilldown  # noqa: E402

OUTDIR = os.path.join(REPO, "analysis", "_anomaly_step3")


def panel(ax, meta, res):
    xs = res["xs"]
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.fill_between(xs, res["env_lo"], res["env_hi"], color="#4C78A8", alpha=0.2, lw=0, zorder=2)
    ax.plot(xs, res["det_y"], color="#4C78A8", lw=1.5, zorder=3)
    ax.errorbar(res["rho"], res["P_mean"], yerr=res["P_err"], fmt="o", ms=3.5,
                color="#E45756", ecolor="#E45756", elinewidth=1, capsize=2, zorder=4)
    ps = res["rho_star_mean"] > 0
    if ps:
        ax.axvline(res["rho_star_mean"], color="#54A24B", ls="--", lw=1.1, zorder=3)
    ax.set_title(
        f"{meta['model']} row{meta['row']} [{meta['verdict']}]\n"
        f"sep_frac={res['sep_frac']:.2f}  rho*={res['rho_star_mean']:.2f}  "
        f"diff={meta['diff_density']:.2f}  lbl={meta['label_density']:.2f}",
        fontsize=7.5)
    ax.set_xlabel(r"$\rho$", fontsize=7)
    ax.set_ylabel("P", fontsize=7)
    ax.tick_params(labelsize=6)
    pm = res["P_mean"]; neg = pm[pm < 0]
    lo = neg.min() * 1.3 if neg.size else -1
    ax.set_ylim(lo, max(5.0, abs(lo) * 0.6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=os.path.join(OUTDIR, "eos_diff_report.csv"))
    ap.add_argument("--verdicts", default="FALSE_NEG,FLAG_MISMATCH,FALSE_POS")
    ap.add_argument("--nboot", type=int, default=1000)
    args = ap.parse_args()

    rep = pd.read_csv(args.report)
    want = set(args.verdicts.split(","))
    rep = rep[rep["verdict"].isin(want)]

    rows = []
    per_model = {}
    for _, r in rep.iterrows():
        model, row = r["model"], int(r["row"])
        d = os.path.join(REPO, "runs", model, "anomalies", f"poly{row}")
        if not os.path.isdir(d):
            continue
        try:
            P, err, rho = get_EOS(d, frac=0.5, bootstrap=True)
            if not rho:
                continue
            res = drilldown(rho, P, nboot=args.nboot, seed=12345)
        except Exception as e:
            print(f"  skip {model} row{row}: {e}", flush=True)
            continue
        meta = dict(model=model, row=row, verdict=r["verdict"],
                    diff_density=float(r.get("diff_density", np.nan)),
                    label_density=float(r.get("label_density", np.nan)))
        per_model.setdefault(model, []).append((meta, res))
        rows.append(dict(
            model=model, row=row, verdict=r["verdict"],
            label_density=meta["label_density"], label_exp=float(r.get("label_exp_density", np.nan)),
            report_diff_density=meta["diff_density"],
            eos_sep_frac=round(res["sep_frac"], 3),
            eos_rho_star=round(res["rho_star_mean"], 3),
            eos_rho_star_std=round(res["rho_star_std"], 3),
            eos_exp_density=round(res["exp_density_mean"], 3),
            ps_by_sepfrac=bool(res["sep_frac"] >= 0.5),
        ))
        print(f"  {model:9s} row{row:4d} {r['verdict']:14s} sep_frac={res['sep_frac']:.2f} "
              f"rho*={res['rho_star_mean']:.2f} diff={meta['diff_density']:.2f}", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    for model, items in per_model.items():
        n = len(items); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (meta, res) in zip(axes.flat, items):
            ax.axis("on"); panel(ax, meta, res)
        fig.suptitle(f"{model} — flagged-sequence P(rho) drill-down", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = os.path.join(OUTDIR, f"loop_drilldown_{model}.png")
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"wrote {out}", flush=True)

    if rows:
        df = pd.DataFrame(rows).sort_values(["model", "row"])
        csv = os.path.join(OUTDIR, "loop_drilldown_summary.csv")
        df.to_csv(csv, index=False)
        print(f"wrote {csv}  ({len(df)} sequences)")
        # how many flagged FALSE_NEG actually clear sep_frac>=0.5 ?
        fn = df[df["verdict"] == "FALSE_NEG"]
        if len(fn):
            print(f"\nFALSE_NEG: {int(fn['ps_by_sepfrac'].sum())}/{len(fn)} clear sep_frac>=0.5 "
                  f"(the rest are shallow/marginal loops)")
    else:
        print("No flagged EoS sweeps found locally yet — copy them first "
              "(analysis/_anomaly_step3/pull_loop_eos.sh + rsync into runs/).")


if __name__ == "__main__":
    main()
