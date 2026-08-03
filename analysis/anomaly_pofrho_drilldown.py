"""Step 3 P(rho) drill-down for the anomalous-sequence analysis.

Re-runs the EoS caller on the flagged sequences whose trajectories were copied
into ``runs/<MODEL>/anomalies/poly<global_idx>/`` and captures the diagnostics
that ``process_eos_sims.bootstrap_eos_analysis`` computes but does not return
(``sep_frac``, the bootstrap pressure envelope, the highest-root distribution).

Faithful to ``bootstrap_eos_analysis``: same 5-block split into independent
pressures (via ``get_EOS(bootstrap=True)``), same anchored cubic spline
(``bc_type=((1,0),(2,0))`` through (0,0)), same ``near_flags`` significance
test, same ``threshold_fraction=0.5`` non-PS rule, same ``work=15`` expenditure
integral. Inner loops are vectorised for speed but the numerics are unchanged.

Does NOT modify ``process_eos_sims.py`` (non-goal in the plan). Reuses
``get_EOS`` and ``calc_exp_density`` from it directly.

Outputs (under ``analysis/_anomaly_step3/``):
  - ``<MODEL>_pofrho.png`` : one P(rho) panel per flagged poly for the model.
  - ``drilldown_summary.csv`` : one row per poly with the captured diagnostics.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root_scalar

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.process_eos_sims import get_EOS  # noqa: E402


def calc_exp_density(spline, rhomin, rhomax, work=15):
    """Vectorised, numerically identical twin of
    ``process_eos_sims.calc_exp_density`` (same x-grid, same cumulative
    trapezoid, same first-crossing return). Kept local so the 1000-resample
    bootstrap runs in seconds rather than minutes; the pipeline function is
    left untouched.
    """
    if rhomax > 1.6:
        x = np.linspace(0.000001, rhomax + 0.4, 10000)
    else:
        x = np.linspace(0.000001, rhomax + 0.2, 10000)
    y = spline(x) / x ** 2
    cum = cumulative_trapezoid(y, x, initial=0.0)
    hit = np.argmax(cum > work)
    if cum[hit] > work:
        return float(x[hit])
    return -1.0

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_anomaly_step3")

# Flagged polys copied to runs/<MODEL>/anomalies/, keyed by global row index.
# Fields: (poly, subgroup, blurb, campaign_density, campaign_exp, campaign_diff)
# from labels_gen10.csv. campaign_density == 0 marks an anomaly (the flag rule);
# campaign_density > 0 rows are the rest of the homopolymer family, uploaded for
# context but NOT flagged under that force field (they act as PS controls).
POLYS = {
    "CALVADOS": [
        ("poly63", "seed", "poly-Y L36", 1.258, 0.255, 0.3270),
        ("poly68", "seed", "poly-W L143", 0.000, 1.791, 0.0353),
        ("poly72", "seed", "poly-W L21", 1.296, 0.310, 0.6598),
        ("poly83", "seed", "poly-Y L156", 0.000, 1.692, 0.0317),
        ("poly85", "seed", "poly-Y L47", 1.265, 0.470, 0.2305),
        ("poly118", "seed", "poly-W L46", 0.000, 1.652, 0.2126),
    ],
    "MPIPI": [
        ("poly68", "seed", "poly-W L143", 0.000, 1.757, 0.00079),
        ("poly174", "max-max g2", "RY-repeat + E18 block", 0.000, 1.748, 0.0109),
        ("poly277", "max-max g4", "poly-H/W + E-block", 0.000, 1.216, 0.0488),
        ("poly331", "max-max g5", "YR/D block copolymer", 0.000, 1.160, 0.0711),
    ],
    "HPS_URRY": [
        ("poly68", "seed", "poly-W L143", 0.000, 1.336, 0.0400),
    ],
}


def drilldown(rho, P, nboot=1000, work=15, threshold_fraction=0.5, seed=0):
    """Faithful re-run of bootstrap_eos_analysis returning the diagnostics.

    Parameters
    ----------
    rho : list of float
        Simulated densities (already sorted, from get_EOS).
    P : list of array
        Per-density independent pressure samples (the 5 block means).
    nboot : int
        Bootstrap resamples.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        rho, P_mean, P_err, xs, env_mean, env_lo, env_hi, det_spline_y,
        sep_frac, rho_star_mean, rho_star_std, exp_density_mean,
        exp_density_std, n_cond, det_root.
    """
    rng = np.random.default_rng(seed)
    rho = np.asarray(rho, float)
    P_mean = np.array([np.mean(pv) for pv in P])
    P_err = np.array([np.std(pv, ddof=1) / np.sqrt(len(pv)) for pv in P])

    rhos = np.insert(rho, 0, 0.0)
    xs = np.linspace(0.05, rhos.max() + 0.2, 2000)

    Pb = np.empty((nboot, xs.size))
    splines = []
    exp_vals = np.empty(nboot)
    for i in range(nboot):
        Pboot = np.array(
            [np.mean(rng.choice(pv, size=len(pv), replace=True)) for pv in P]
        )
        Pboot = np.insert(Pboot, 0, 0.0)
        S = CubicSpline(rhos, Pboot, bc_type=((1, 0.0), (2, 0.0)))
        splines.append(S)
        Pb[i] = S(xs)
        exp_vals[i] = calc_exp_density(S, rho.min(), rhos.max(), work=work)

    sd = Pb.std(axis=0)
    eps = 1.96 * sd
    near = (Pb <= eps).any(axis=1)  # shape (nboot,)

    cond = []
    for b, S in enumerate(splines):
        if not near[b]:
            continue
        vals = Pb[b]
        idx = np.where(vals[:-1] * vals[1:] < 0)[0]
        roots = list(xs[np.where(vals == 0)[0]])
        for k in idx:
            try:
                roots.append(
                    root_scalar(S, bracket=[xs[k], xs[k + 1]], method="brentq").root
                )
            except Exception:
                pass
        rr = [r for r in roots if r > rho.min()]
        if rr:
            rstar = max(rr)
            if np.isfinite(rstar):
                cond.append(rstar)

    sep_frac = float(near.mean())
    if sep_frac >= threshold_fraction and cond:
        rho_star_mean = float(np.mean(cond))
        rho_star_std = float(np.std(cond, ddof=1)) if len(cond) > 1 else 0.0
    else:
        rho_star_mean = 0.0
        rho_star_std = 0.0

    ev = exp_vals[exp_vals > 0]
    exp_density_mean = float(np.mean(ev)) if ev.size else -1.0
    exp_density_std = float(np.std(ev)) if ev.size else -1.0

    # Deterministic spline through the mean pressures, for the overlay + a
    # single-shot highest root (independent of the significance gate).
    det = CubicSpline(rhos, np.insert(P_mean, 0, 0.0), bc_type=((1, 0.0), (2, 0.0)))
    dv = det(xs)
    didx = np.where(dv[:-1] * dv[1:] < 0)[0]
    droots = []
    for k in didx:
        try:
            droots.append(
                root_scalar(det, bracket=[xs[k], xs[k + 1]], method="brentq").root
            )
        except Exception:
            pass
    droots = [r for r in droots if r > rho.min()]
    det_root = max(droots) if droots else np.nan

    return dict(
        rho=rho,
        P_mean=P_mean,
        P_err=P_err,
        xs=xs,
        env_mean=Pb.mean(axis=0),
        env_lo=np.percentile(Pb, 2.5, axis=0),
        env_hi=np.percentile(Pb, 97.5, axis=0),
        det_y=dv,
        sep_frac=sep_frac,
        rho_star_mean=rho_star_mean,
        rho_star_std=rho_star_std,
        exp_density_mean=exp_density_mean,
        exp_density_std=exp_density_std,
        n_cond=len(cond),
        det_root=det_root,
    )


def classify(res, campaign_density, campaign_exp):
    """Assign a qualitative verdict to a drill-down result.

    Compares the re-run against the campaign label so the verdict distinguishes
    genuine anomalies (campaign density==0) from the PS controls, and separates
    stale-caller false-negatives from grid-truncated uploads.

    A trustworthy rho* must be backed by an actual positive-pressure branch in
    the simulated grid (``n_pos > 0``). When every simulated pressure is
    negative the caller can still return a nonzero rho* from the spline's
    linear extrapolation beyond rho_max -- a spurious boundary root, not a real
    condensate (exp_density comes back -1 in that case).
    """
    n_pos = int(np.sum(res["P_mean"] > 0))
    rerun_ps = res["rho_star_mean"] > 0
    if campaign_density > 0:
        return "control: PS in campaign (not an anomaly)"
    # campaign called it non-PS (density==0) -> a flagged anomaly
    if n_pos > 0 and rerun_ps:
        return "false-negative: stale non-PS call, current caller -> PS"
    # no positive branch in the uploaded grid; campaign exp>0 says the true
    # repulsive branch (and rho*) sits beyond the uploaded rho_max
    return "grid-truncated upload: rho* beyond rho_max (grid-extent FN)"


def make_panel(ax, name, blurb, res, campaign_density):
    xs = res["xs"]
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.fill_between(
        xs, res["env_lo"], res["env_hi"], color="#4C78A8", alpha=0.20,
        lw=0, zorder=2, label="bootstrap 95%",
    )
    ax.plot(xs, res["det_y"], color="#4C78A8", lw=1.6, zorder=3, label="spline (mean P)")
    ax.errorbar(
        res["rho"], res["P_mean"], yerr=res["P_err"], fmt="o", ms=4,
        color="#E45756", ecolor="#E45756", elinewidth=1, capsize=2,
        zorder=4, label="sim P(rho)",
    )
    n_pos = int(np.sum(res["P_mean"] > 0))
    real_rho_star = res["rho_star_mean"] > 0 and n_pos > 0
    if real_rho_star:
        ax.axvline(res["rho_star_mean"], color="#54A24B", ls="--", lw=1.2, zorder=3)
    camp = f"PS (rho={campaign_density:.2f})" if campaign_density > 0 else "non-PS (rho=0)"
    if real_rho_star:
        rerun = f"rho*={res['rho_star_mean']:.2f}"
    elif n_pos == 0:
        rerun = "P<0 throughout: rho* beyond grid"
    else:
        rerun = "no rho* on grid"
    ax.set_title(
        f"{name}  ({blurb})\n"
        f"campaign: {camp}  ->  rerun: {rerun}\n"
        f"sep_frac={res['sep_frac']:.2f}   exp_d={res['exp_density_mean']:.2f}",
        fontsize=8,
    )
    ax.set_xlabel(r"$\rho$  (g/mL)", fontsize=8)
    ax.set_ylabel(r"$P$", fontsize=8)
    ax.tick_params(labelsize=7)
    # keep the loop visible: clip the huge high-density repulsive branch
    pm = res["P_mean"]
    neg = pm[pm < 0]
    lo = neg.min() * 1.3 if neg.size else -1
    hi = max(5.0, abs(lo) * 0.6)
    ax.set_ylim(lo, hi)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for model, plist in POLYS.items():
        n = len(plist)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for j, (poly, subgroup, blurb, camp_dens, camp_exp, camp_diff) in enumerate(plist):
            pth = os.path.join(REPO, "runs", model, "anomalies", poly)
            P, err, rho = get_EOS(pth, frac=0.5, bootstrap=True)
            res = drilldown(rho, P, nboot=1000, seed=12345)
            verdict = classify(res, camp_dens, camp_exp)
            ax = axes.flat[j]
            ax.axis("on")
            make_panel(ax, poly, blurb, res, camp_dens)
            rows.append(
                dict(
                    model=model,
                    poly=poly,
                    subgroup=subgroup,
                    description=blurb,
                    n_rho=len(rho),
                    rho_max=float(max(rho)),
                    P_min=float(np.min(res["P_mean"])),
                    P_max=float(np.max(res["P_mean"])),
                    n_pos=int(np.sum(res["P_mean"] > 0)),
                    sep_frac=round(res["sep_frac"], 3),
                    campaign_density=camp_dens,
                    rerun_rho_star=round(res["rho_star_mean"], 3),
                    rerun_rho_star_std=round(res["rho_star_std"], 3),
                    campaign_exp_density=camp_exp,
                    rerun_exp_density=round(res["exp_density_mean"], 3),
                    campaign_diff=camp_diff,
                    verdict=verdict,
                )
            )
            print(
                f"{model:9s} {poly:8s} sep={res['sep_frac']:.2f} "
                f"camp_rho={camp_dens:.2f} rerun_rho*={res['rho_star_mean']:.2f} "
                f"exp: camp={camp_exp:.2f}/rerun={res['exp_density_mean']:.2f} -> {verdict}",
                flush=True,
            )
        # shared legend
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(f"{model} — anomaly P(rho) drill-down", fontsize=11)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        out = os.path.join(OUTDIR, f"{model}_pofrho.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out}", flush=True)

    df = pd.DataFrame(rows)
    csv = os.path.join(OUTDIR, "drilldown_summary.csv")
    df.to_csv(csv, index=False)
    print(f"wrote {csv}", flush=True)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
