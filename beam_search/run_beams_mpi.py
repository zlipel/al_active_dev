import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from mpi4py import MPI
from time import time
 
from cross_paths.model_io import load_all_models
from cross_paths.beam_search import beam_search_paths
from beam_search.policy import BeamPolicy
from al_pipeline.core.paths import ALPaths
import torch
import numba as nb
import cProfile
import pstats
import io as _sysio

WORKTAG = 1
DIETAG = 2

KEY_COLS = ["start_idx", "du_req", "dv_req"]

FINAL_REASONS = {
    "outside_hull",
    "finished_quantile",
    "no_finished",
    "no_valid_candidates",
}

def endpoint_key(row):
    return (int(row.start_idx), round(float(row.du_req), 8), round(float(row.dv_req), 8))


def result_csv_for_start(paths_dir, start_idx):
    return os.path.join(paths_dir, "RESULTS", f"start_{int(start_idx):04d}", "paths.csv")

def load_existing_results(paths_dir, start_idx):
    out_csv = result_csv_for_start(paths_dir, start_idx)
    if not os.path.exists(out_csv):
        return None
    try:
        return pd.read_csv(out_csv)
    except Exception as e:
        print(f"[resume] could not read {out_csv}: {e}", flush=True)
        return None

def build_existing_reason_map(existing_df):
    if existing_df is None or existing_df.empty:
        return {}
    needed = set(KEY_COLS + ["reason"])
    if not needed.issubset(existing_df.columns):
        return {}
    out = {}
    for row in existing_df.itertuples(index=False):
        out[endpoint_key(row)] = row.reason
    return out


def get_pending_start_indices(all_start_indices, groups_by_start, paths_dir, resume, extend_no_finished):
    """
    Return (pending_list, n_with_results, n_skipped).

    When resume=False returns (list(all_start_indices), 0, 0).
    When resume=True, a start is skipped only if every expected endpoint key has
    a final reason recorded; n_with_results counts starts that have any existing
    results file, n_skipped counts those fully complete.
    """
    if not resume:
        return list(all_start_indices), 0, 0

    pending = []
    n_with_results = 0
    n_skipped = 0

    for start_idx in all_start_indices:
        sub = groups_by_start[int(start_idx)]
        expected_keys = {endpoint_key(r) for r in sub[KEY_COLS].itertuples(index=False)}

        existing_df = load_existing_results(paths_dir, start_idx)
        reason_map  = build_existing_reason_map(existing_df)

        if reason_map:
            n_with_results += 1

        done_keys = set()
        for k, reason in reason_map.items():
            if reason in FINAL_REASONS:
                if extend_no_finished and reason == "no_finished":
                    continue
                done_keys.add(k)

        if done_keys >= expected_keys:
            n_skipped += 1
            continue

        pending.append(start_idx)

    return pending, n_with_results, n_skipped


def handoutWork(start_indices, comm, numProcesses):
    totalWork = len(start_indices)
    workcount = 0
    recvcount = 0
    print("conductor sending first tasks", flush=True)
    for idx in range(1, numProcesses):
        if workcount < totalWork:
            work = start_indices[workcount]
            comm.send(work, dest=idx, tag=WORKTAG)
            workcount += 1
            print(f"conductor sent {work} to {idx}", flush=True)
        else:
            comm.send(-1, dest=idx, tag=DIETAG)

    while workcount < totalWork:
        stat = MPI.Status()
        start_idx = comm.recv(source=MPI.ANY_SOURCE, status=stat)
        recvcount += 1
        workerId = stat.Get_source()
        print(f"conductor received {start_idx} from {workerId}", flush=True)
        work = int(start_indices[workcount])
        comm.send(work, dest=workerId, tag=WORKTAG)
        workcount += 1
        print(f"conductor sent {work} to {workerId}", flush=True)

    while recvcount < totalWork:
        stat = MPI.Status()
        start_idx = comm.recv(source=MPI.ANY_SOURCE, status=stat)
        recvcount += 1
        workerId = stat.Get_source()
        print(f"end: conductor received {start_idx} from {workerId}", flush=True)

    for idx in range(1, numProcesses):
        comm.send(-1, dest=idx, tag=DIETAG)


def _build_policy(bundle, q_rho, q_diff, *, kind, start_regime,
                  hard_threshold, reject_threshold, feat_threads):
    """Construct a `BeamPolicy` for one start.

    Called once per start inside `doWork` so ``start_regime`` can vary
    across the AL start pool. Cheap — `BeamPolicy.__init__` only wraps
    references to the existing bundle + transforms.
    """
    return BeamPolicy(
        kind=kind,
        surrogate=bundle.surrogate,
        featurizer=bundle.featurizer,
        q_rho=q_rho,
        q_diff=q_diff,
        start_regime=start_regime,
        hard_threshold=hard_threshold,
        reject_threshold=reject_threshold,
        feat_threads=feat_threads,
    )


def worker(comm, groups_by_start, paths_dir, bundles, model, q_rho, q_diff,
           quantile_tol, feat_threads, beam_width, max_steps, length_changes,
           policy_kind, hard_threshold, reject_threshold,
           resume=False, extend_no_finished=False, extra_steps=0,
           patience=0, min_delta=0.0,
           profile_rank=-1):
    _profiled = False

    bundle = bundles[model]
    dummy = [bundle.sequences[0]]

    # warm up numba featurizer + GP posterior. Use a synthetic PS start so
    # expert_tied/anchored_reject construction is valid during warm-up; the
    # policy's own predict_candidates path warms up both featurizer and GP.
    warm_policy = _build_policy(
        bundle, q_rho, q_diff,
        kind=policy_kind,
        start_regime="ps" if policy_kind in ("expert_tied", "anchored_reject") else None,
        hard_threshold=hard_threshold,
        reject_threshold=reject_threshold,
        feat_threads=feat_threads,
    )
    _ = warm_policy.predict_candidates(dummy)
    while True:
        stat = MPI.Status()
        start_idx = comm.recv(source=0, tag=MPI.ANY_TAG, status=stat)
        print(f"worker {comm.Get_rank()} got {start_idx}", flush=True)
        if stat.Get_tag() == DIETAG:
            print(f"worker {comm.Get_rank()} dying", flush=True)
            return
        start_time = time()
        nb.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        print(f"[rank {comm.Get_rank()}] after set: numba={nb.get_num_threads()} layer={nb.threading_layer()}", flush=True)
        do_kwargs = dict(
            groups_by_start=groups_by_start,
            paths_dir=paths_dir,
            bundle=bundle,
            model=model,
            q_rho=q_rho,
            q_diff=q_diff,
            quantile_tol=quantile_tol,
            feat_threads=feat_threads,
            beam_width=beam_width,
            max_steps=max_steps,
            length_changes=length_changes,
            policy_kind=policy_kind,
            hard_threshold=hard_threshold,
            reject_threshold=reject_threshold,
            resume=resume,
            extend_no_finished=extend_no_finished,
            extra_steps=extra_steps,
            patience=patience,
            min_delta=min_delta,
        )
        if profile_rank == comm.Get_rank() and not _profiled:
            pr = cProfile.Profile()
            pr.enable()
            doWork(int(start_idx), **do_kwargs)
            pr.disable()
            _profiled = True
            s = _sysio.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(40)
            profile_path = os.path.join(paths_dir, f"profile_rank{comm.Get_rank():03d}.txt")
            with open(profile_path, "w") as f:
                f.write(s.getvalue())
            print(f"[rank {comm.Get_rank()}] cProfile written to {profile_path}", flush=True)
        else:
            doWork(int(start_idx), **do_kwargs)
        elapsed = time() - start_time
        hrs = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        sec = int(elapsed % 60)
        print(f"worker {comm.Get_rank()} finished {start_idx} in {hrs:02}:{mins:02}:{sec:02}", flush=True)
        comm.send(start_idx, dest=0)


def doWork(start_idx, groups_by_start, paths_dir, bundle, model, q_rho, q_diff,
           quantile_tol, feat_threads, beam_width, max_steps, length_changes,
           policy_kind, hard_threshold, reject_threshold,
           resume=False, extend_no_finished=False, extra_steps=0,
           patience=0, min_delta=0.0):
    if int(start_idx) not in groups_by_start:
        print(f"[rank {MPI.COMM_WORLD.Get_rank()}] start_idx={start_idx} not found; skipping", flush=True)
        return

    sub = groups_by_start[int(start_idx)].copy()

    out_dir = os.path.join(paths_dir, "RESULTS", f"start_{start_idx:04d}")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "paths.csv")

    existing_df = load_existing_results(paths_dir, start_idx) if resume else None
    reason_map = build_existing_reason_map(existing_df)

    if resume:
        keep_mask = []
        for row in sub.itertuples(index=False):
            prev_reason = reason_map.get(endpoint_key(row), None)

            if prev_reason is None:
                keep_mask.append(True)
            elif extend_no_finished and prev_reason == "no_finished":
                keep_mask.append(True)
            else:
                keep_mask.append(False)

        sub = sub.loc[keep_mask].copy()

        if sub.empty:
            print(f"[rank {MPI.COMM_WORLD.Get_rank()}] start_idx={start_idx} already complete; skipping", flush=True)
            return

    rho_start = float(sub["rho_start"].iloc[0])
    diff_start = float(sub["diff_start"].iloc[0])
    u_start = float(sub["u_start"].iloc[0])
    v_start = float(sub["v_start"].iloc[0])

    # Row 8 wrote `start_regime` (ps/nonps) into every endpoint row per §III.8.
    # All rows for this start_idx share it — read once from the first row.
    start_regime = None
    if "start_regime" in sub.columns:
        start_regime = str(sub["start_regime"].iloc[0])

    policy = _build_policy(
        bundle, q_rho, q_diff,
        kind=policy_kind,
        start_regime=start_regime,
        hard_threshold=hard_threshold,
        reject_threshold=reject_threshold,
        feat_threads=feat_threads,
    )

    s_rho, s_diff = policy.label_scalers
    z_rho_start = float(s_rho.transform([[rho_start]])[0, 0])
    z_diff_start = float(s_diff.transform([[diff_start]])[0, 0])

    start_phys = np.array([rho_start, diff_start], dtype=float)
    start_uv = np.array([u_start, v_start], dtype=float)
    start_z = np.array([z_rho_start, z_diff_start], dtype=float)

    rows_out = []
    n_ep = len(sub)
    _rank = MPI.COMM_WORLD.Get_rank()

    for ep_idx, row in enumerate(sub.itertuples(index=False)):
        d = row._asdict()

        prev_reason = reason_map.get(endpoint_key(row), None)

        row_max_steps = max_steps
        if extend_no_finished and prev_reason == "no_finished":
            row_max_steps = max_steps + extra_steps

        if not row.inside_hull:
            d.update(dict(
                attempted=False,
                hit=False,
                reason="outside_hull",
                n_edits=np.nan,
                path_len=np.nan,
                rho_target=np.nan,
                diff_target=np.nan,
                z_rho_target=np.nan,
                z_diff_target=np.nan,
                rho_end=np.nan,
                diff_end=np.nan,
                z_rho_end=np.nan,
                z_diff_end=np.nan,
                u_end=np.nan,
                v_end=np.nan,
                du_ach=np.nan,
                dv_ach=np.nan,
                drho=np.nan,
                ddiff=np.nan,
                end_seq=None,
            ))
            rows_out.append(d)
            continue

        u_t = float(row.u_target)
        v_t = float(row.v_target)
        rho_t = float(q_rho.inverse_transform([[u_t]])[0, 0])
        diff_t = float(q_diff.inverse_transform([[v_t]])[0, 0])

        #_t_ep = time()
        finished, beam_tail = beam_search_paths(
            policy=policy,
            start_seq=row.start_seq,
            uv_target=np.array([u_t, v_t], dtype=float),
            start_phys=start_phys,
            start_uv=start_uv,
            start_z=start_z,
            beam_width=beam_width,
            max_steps=row_max_steps,
            tol=quantile_tol,
            diversity="none",
            diversity_radius=0.0,
            budget_cost=None,
            length_changes=length_changes,
            axis_weights=(1.0, 1.0),
            min_positive=1e-12,
            patience=patience,
            min_delta=min_delta,
        )
        #_ep_elapsed = time() - _t_ep
        #_result = "finished" if finished else ("no_finished" if beam_tail else "no_valid")
        #print(f"[TIMING endpoint] rank={_rank} start_idx={start_idx} ep={ep_idx+1}/{n_ep} L={len(row.start_seq)} elapsed={_ep_elapsed:.2f}s result={_result}", flush=True)

        if finished:
            best = finished[0]
            hit = True
            reason = "finished_quantile"
        elif beam_tail:
            best = beam_tail[0]
            hit = False
            reason = "no_finished"
        else:
            d.update(dict(
                attempted=True,
                hit=False,
                reason="no_valid_candidates",
                n_edits=np.nan,
                path_len=np.nan,
                rho_target=rho_t,
                diff_target=diff_t,
                z_rho_target=np.nan,
                z_diff_target=np.nan,
                rho_end=np.nan,
                diff_end=np.nan,
                z_rho_end=np.nan,
                z_diff_end=np.nan,
                u_end=np.nan,
                v_end=np.nan,
                du_ach=np.nan,
                dv_ach=np.nan,
                drho=np.nan,
                ddiff=np.nan,
                end_seq=None,
            ))
            rows_out.append(d)
            continue

        z_rho_end, z_diff_end = best["preds_z"][-1]
        rho_end, diff_end = best["preds_phys"][-1]
        u_end, v_end = best["preds_uv"][-1]

        z_rho_t = float(s_rho.transform([[rho_t]])[0, 0])
        z_diff_t = float(s_diff.transform([[diff_t]])[0, 0])

        d.update(dict(
            attempted=True,
            hit=hit,
            reason=reason,
            n_edits=len(best["edits"]),
            path_len=len(best["path"]),
            rho_target=rho_t,
            diff_target=diff_t,
            z_rho_target=z_rho_t,
            z_diff_target=z_diff_t,
            rho_end=float(rho_end),
            diff_end=float(diff_end),
            z_rho_end=float(z_rho_end),
            z_diff_end=float(z_diff_end),
            u_end=float(u_end),
            v_end=float(v_end),
            du_ach=float(u_end - u_start),
            dv_ach=float(v_end - v_start),
            drho=float(rho_end - rho_start),
            ddiff=float(diff_end - diff_start),
            end_seq=best["path"][-1],
            endpoint_p_ps=float(best.get("endpoint_p_ps", float("nan"))),
        ))
        rows_out.append(d)

    new_df = pd.DataFrame(rows_out)

    order_df = groups_by_start[int(start_idx)][KEY_COLS].drop_duplicates().copy()
    order_df["_order"] = np.arange(len(order_df))

    if existing_df is not None and not existing_df.empty:
        # Strip stale rows whose keys are no longer in the current endpoint grid.
        existing_df = existing_df.merge(order_df[KEY_COLS], on=KEY_COLS, how="inner")
        out_df = pd.concat([existing_df, new_df], ignore_index=True)
        out_df = out_df.drop_duplicates(subset=KEY_COLS, keep="last")
    else:
        out_df = new_df.drop_duplicates(subset=KEY_COLS, keep="last")

    # Current endpoint grid is the source of truth: exactly one row per expected endpoint.
    out_df = order_df.merge(out_df, on=KEY_COLS, how="left")
    out_df = out_df.sort_values("_order").drop(columns="_order")

    out_df.to_csv(out_csv, index=False)
    print(f"[rank {MPI.COMM_WORLD.Get_rank()}] start_idx={start_idx} done; results written to {out_csv}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch_dir", required=True)
    parser.add_argument("--home_dir", required=True)
    parser.add_argument("--db_root", required=True)
    parser.add_argument("--model", required=True, choices=["HPS_URRY", "MPIPI", "CALVADOS"])
    parser.add_argument("--final_iter", type=int, default=8)
    parser.add_argument("--feat_threads", type=int, default=12)
    parser.add_argument("--torch_threads", type=int, default=12)
    parser.add_argument("--beam_width", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--tol_u", type=float, default=0.05)
    parser.add_argument("--tol_v", type=float, default=0.05)
    parser.add_argument("--length_changes", action='store_true')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend_no_finished", action="store_true",
                        help="Rerun endpoints whose prior result was no_finished")
    parser.add_argument("--extra_steps", type=int, default=0,
                        help="Additional steps to allow when extending no_finished endpoints")
    parser.add_argument("--stagnation_patience", type=int, default=0,
                        help="Stop after this many non-improving beam steps; 0 disables")
    parser.add_argument("--stagnation_delta", type=float, default=0.0,
                        help="Minimum improvement in best distance to reset stagnation")
    parser.add_argument("--profile_rank", type=int, default=-1,
                        help="Wrap the first doWork call on this rank with cProfile (use 1 for first worker)")
    # ALPaths construction args — needed to resolve GPR checkpoint with the correct naming
    parser.add_argument("--front",                default="upper",
                        help="Pareto front ('upper' or 'lower')")
    parser.add_argument("--ehvi_variant",         default="epsilon",
                        help="EHVI variant (default: epsilon)")
    parser.add_argument("--exploration_strategy", default="kriging_believer",
                        help="Exploration strategy (default: kriging_believer)")
    parser.add_argument("--transform",            default="yeoj",
                        help="Label transform (default: yeoj)")
    parser.add_argument("--mc_ehvi", action="store_true",
                        help="Use MC-EHVI checkpoint naming")
    # --- Row 9: beam policy ---
    parser.add_argument(
        "--policy",
        choices=["expert_tied", "anchored_reject", "soft", "hard", "global"],
        default="expert_tied",
        help=(
            "Beam-search policy (Row 9). expert_tied is the diagnostic "
            "primary; others are reachable but not analyzed this branch."
        ),
    )
    parser.add_argument("--hard_threshold", type=float, default=0.5,
                        help="Gate threshold for --policy hard")
    parser.add_argument("--reject_threshold", type=float, default=0.5,
                        help="Gate threshold for --policy anchored_reject "
                             "(candidates confidently opposite-regime are rejected)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    model = args.model

    print(f"[rank {rank}] Starting run_beams_mpi for model={model} policy={args.policy} length_changes={args.length_changes}", flush=True)

    # Endpoint CSVs are policy-agnostic (same start × target grid feeds every
    # policy); result CSVs are policy-scoped so runs don't cross-contaminate
    # (§IV.output-layout of the plan).
    endpoints_root = os.path.join(
        args.scratch_dir,
        "PATHS" if args.length_changes else "PATHS_FIXED_LENGTH",
        model,
    )
    paths_dir = os.path.join(endpoints_root, args.policy)
    os.makedirs(paths_dir, exist_ok=True)

    if rank == 0:
        endpoints_csv = os.path.join(endpoints_root, f"endpoints_{args.model}.csv")
        print(f"[rank 0] Loading endpoints from {endpoints_csv}", flush=True)
        endpoints_all = pd.read_csv(endpoints_csv)
    else:
        endpoints_all = None

    endpoints_all = comm.bcast(endpoints_all, root=0)
    lens = endpoints_all.drop_duplicates("start_idx").set_index("start_idx")["start_seq"].str.len()
    groups_by_start = {int(k): v for k, v in endpoints_all.groupby("start_idx")}

    all_start_indices = sorted(lens.index.tolist(), key=lambda idx: lens.loc[idx], reverse=True)

    start_indices, n_with_results, n_skipped = get_pending_start_indices(
        all_start_indices, groups_by_start, paths_dir,
        args.resume, args.extend_no_finished,
    )

    if rank == 0 and args.resume:
        n_total   = len(all_start_indices)
        n_pending = len(start_indices)
        print(f"[rank 0] Resume summary:", flush=True)
        print(f"  total starts in endpoints CSV  : {n_total}", flush=True)
        print(f"  starts with existing RESULTS   : {n_with_results}", flush=True)
        print(f"  starts fully complete (skipped): {n_skipped}", flush=True)
        print(f"  starts pending (to process)    : {n_pending}", flush=True)
        first_shown = start_indices[:20]
        print(f"  first {len(first_shown)} pending indices      : {first_shown}", flush=True)

    al_paths = ALPaths(
        base_path=args.home_dir,
        scratch_path=args.scratch_dir,
        iteration=args.final_iter,
        front=args.front,
        model=model,
        ehvi_variant=args.ehvi_variant,
        exploration_strategy=args.exploration_strategy,
        transform=args.transform,
        mc_ehvi=args.mc_ehvi,
    )
    bundles = load_all_models(al_paths, db_dir=os.path.join(args.db_root, "databases"))
    bundle = bundles[model]
    rho = bundle.labels_exp_density
    diff = bundle.labels_diff

    q_rho = QuantileTransformer(n_quantiles=min(1000, len(rho)), random_state=0, output_distribution="uniform").fit(rho.reshape(-1, 1))
    q_diff = QuantileTransformer(n_quantiles=min(1000, len(diff)), random_state=0, output_distribution="uniform").fit(diff.reshape(-1, 1))

    torch.set_num_threads(args.torch_threads)
    
    torch.set_num_interop_threads(1)

    quantile_tol = (args.tol_u, args.tol_v)

    if rank == 0:
        handoutWork(start_indices, comm, size)
    else:
        worker(
            comm, groups_by_start, paths_dir, bundles, model, q_rho, q_diff,
            quantile_tol, args.feat_threads, args.beam_width, args.max_steps, args.length_changes,
            policy_kind=args.policy,
            hard_threshold=args.hard_threshold,
            reject_threshold=args.reject_threshold,
            resume=args.resume,
            extend_no_finished=args.extend_no_finished,
            extra_steps=args.extra_steps,
            patience=args.stagnation_patience,
            min_delta=args.stagnation_delta,
            profile_rank=args.profile_rank
        )

    comm.Barrier()
    if rank == 0:
        print("[rank 0] All ranks finished", flush=True)


if __name__ == "__main__":
    main()
