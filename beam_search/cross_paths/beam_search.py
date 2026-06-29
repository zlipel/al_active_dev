import numpy as np
from time import perf_counter
from mpi4py import MPI
from .model_io import predict_labels_for_sequences

AA = list("ACDEFGHIKLMNPQRSTVWY")


def edit_unit_cost(op: str) -> int:
    return 1 if op == 'sub' else 2


def apply_edit(seq, op, pos, aa=None):
    if op == 'sub':
        return seq[:pos] + aa + seq[pos+1:]
    if op == 'ins':
        return aa + seq if pos == 'start' else seq + aa
    if op == 'del':
        return seq[1:] if pos == 'start' else seq[:-1]
    raise ValueError(op)


def enumerate_neighbors(seq, length_changes=True, sub_palette=None):
    L = len(seq)

    if length_changes:
        if L < 160:
            ins_start_AA = sub_palette.get('start', AA) if sub_palette else AA
            ins_end_AA = sub_palette.get('end', AA) if sub_palette else AA
            for aa in ins_start_AA:
                yield ('ins', 'start', aa, apply_edit(seq, 'ins', 'start', aa))
            for aa in ins_end_AA:
                yield ('ins', 'end', aa, apply_edit(seq, 'ins', 'end', aa))
        if L > 20:
            yield ('del', 'start', None, apply_edit(seq, 'del', 'start'))
            yield ('del', 'end', None, apply_edit(seq, 'del', 'end'))

    for pos, ch in enumerate(seq):
        allowed = sub_palette.get(pos, AA) if sub_palette else AA
        for aa in allowed:
            if aa != ch:
                yield ('sub', pos, aa, apply_edit(seq, 'sub', pos, aa))


# ---------------- prediction helpers ----------------

def predict_candidate_frames(
    bundles: dict,
    model_name: str,
    seq_list,
    q_rho,
    q_diff,
    feat_threads: int = 1,
    min_positive: float = 1e-12,
):
    """
    Predict candidates in the model's native YJ space, then map to physical and
    quantile space for search scoring.

    Returns a dict with keys:
      - z:      (n,2) predicted means in YJ/scaled label space
      - phys:   (n,2) physical (rho_exp, diff)
      - uv:     (n,2) quantile coordinates
      - valid:  (n,) boolean physical-feasibility mask
    """
    bundle = bundles[model_name]
    z = predict_labels_for_sequences(
        bundle,
        seq_list,
        return_std=False,
        batch_size=min(len(seq_list), 4096),
        feat_threads=feat_threads,
    )

    s_rho, s_diff = bundle.label_scalers
    rho_phys = s_rho.inverse_transform(z[:, [0]]).ravel()
    diff_phys = s_diff.inverse_transform(z[:, [1]]).ravel()

    valid = np.isfinite(rho_phys) & np.isfinite(diff_phys) & (rho_phys > min_positive) & (diff_phys > min_positive)

    uv = np.full((len(seq_list), 2), np.nan, dtype=float)
    if np.any(valid):
        uv[valid, 0] = q_rho.transform(rho_phys[valid].reshape(-1, 1)).ravel()
        uv[valid, 1] = q_diff.transform(diff_phys[valid].reshape(-1, 1)).ravel()

    phys = np.column_stack([rho_phys, diff_phys])
    return {"z": z, "phys": phys, "uv": uv, "valid": valid}


def quantile_distance(uv, uv_target, axis_weights=(1.0, 1.0)):
    du = (uv[0] - uv_target[0]) / axis_weights[0]
    dv = (uv[1] - uv_target[1]) / axis_weights[1]
    return float(np.sqrt(du * du + dv * dv))


# ---------------- lightweight node + reconstruction ----------------

def make_node(seq, parent, edit, z, phys, uv, dist, cost, depth, finish_du, finish_dv):
    return {
        "seq": seq,
        "parent": parent,
        "edit": edit,
        "z": z,
        "phys": phys,
        "uv": uv,
        "dist": dist,
        "cost": cost,
        "depth": depth,
        "finish_du": finish_du,
        "finish_dv": finish_dv,
    }


def reconstruct_record(node):
    path = []
    edits = []
    preds_z = []
    preds_phys = []
    preds_uv = []

    cur = node
    while cur is not None:
        path.append(cur["seq"])
        preds_z.append(cur["z"])
        preds_phys.append(cur["phys"])
        preds_uv.append(cur["uv"])
        if cur["edit"] is not None:
            edits.append(cur["edit"])
        cur = cur["parent"]

    path.reverse()
    edits.reverse()
    preds_z.reverse()
    preds_phys.reverse()
    preds_uv.reverse()

    return {
        "path": path,
        "edits": edits,
        "preds_z": np.vstack(preds_z),
        "preds_phys": np.vstack(preds_phys),
        "preds_uv": np.vstack(preds_uv),
        "dist": node["dist"],
        "cost": node["cost"],
        "finish_du": node["finish_du"],
        "finish_dv": node["finish_dv"],
    }


def reconstruct_many(nodes):
    return [reconstruct_record(n) for n in nodes]


# ---------------- search ----------------

def beam_search_paths(
    bundles: dict,
    start_seq: str,
    model_name: str,
    uv_target,
    q_rho,
    q_diff,
    start_phys,
    start_uv,
    start_z=None,
    beam_width: int = 32,
    max_steps: int = 8,
    tol=(0.02, 0.02),
    diversity: str = "none",
    diversity_radius: float = 0.0,
    budget_cost=None,
    feat_threads: int = 1,
    length_changes: bool = True,
    axis_weights=(1.0, 1.0),
    min_positive: float = 1e-12,
    patience: int = 0,
    min_delta: float = 0.0,
):
    uv_target = np.asarray(uv_target, dtype=float)
    phys0 = np.asarray(start_phys, dtype=float)
    uv0 = np.asarray(start_uv, dtype=float)

    if start_z is None:
        bundle = bundles[model_name]
        s_rho, s_diff = bundle.label_scalers
        z0 = np.array([
            float(s_rho.transform([[phys0[0]]])[0, 0]),
            float(s_diff.transform([[phys0[1]]])[0, 0]),
        ], dtype=float)
    else:
        z0 = np.asarray(start_z, dtype=float)

    if not (np.isfinite(phys0[0]) and np.isfinite(phys0[1]) and phys0[0] > min_positive and phys0[1] > min_positive):
        raise ValueError(f"Start sequence {start_seq!r} has invalid provided start_phys={phys0}.")

    start = make_node(
        seq=start_seq,
        parent=None,
        edit=None,
        z=z0,
        phys=phys0,
        uv=uv0,
        dist=quantile_distance(uv0, uv_target, axis_weights=axis_weights),
        cost=0,
        depth=0,
        finish_du=uv0[0] - uv_target[0],
        finish_dv=uv0[1] - uv_target[1],
    )

    finished = []
    beam = [start]
    _rank = MPI.COMM_WORLD.Get_rank()
    best_dist_so_far = np.inf
    stagnant_steps = 0

    for _step in range(max_steps):
        #_L = len(beam[0]["seq"])
        #_t0 = perf_counter()

        cand_meta = []
        for bi, rec in enumerate(beam):
            tip_seq = rec["seq"]
            tip_cost = rec["cost"]
            if budget_cost is not None and tip_cost >= budget_cost:
                continue
            for (op, pos, aa, s2) in enumerate_neighbors(tip_seq, length_changes=length_changes):
                new_cost = tip_cost + edit_unit_cost(op)
                if budget_cost is not None and new_cost > budget_cost:
                    continue
                cand_meta.append((bi, op, pos, aa, s2, new_cost))
        #_t1 = perf_counter()

        if not cand_meta:
            break

        uniq_map = {}
        uniq_list = []
        for _, _, _, _, s2, _ in cand_meta:
            if s2 not in uniq_map:
                uniq_map[s2] = len(uniq_list)
                uniq_list.append(s2)
        #_t2 = perf_counter()

        pred = predict_candidate_frames(
            bundles=bundles,
            model_name=model_name,
            seq_list=uniq_list,
            q_rho=q_rho,
            q_diff=q_diff,
            feat_threads=feat_threads,
            min_positive=min_positive,
        )
        #_t3 = perf_counter()

        expansions = []
        new_finished = []
        for (bi, op, pos, aa, s2, new_cost) in cand_meta:
            k = uniq_map[s2]
            if not pred["valid"][k]:
                continue

            z2 = pred["z"][k]
            phys2 = pred["phys"][k]
            uv2 = pred["uv"][k]
            parent = beam[bi]

            du = uv2[0] - uv_target[0]
            dv = uv2[1] - uv_target[1]

            new_rec = make_node(
                seq=s2,
                parent=parent,
                edit=(op, pos, aa),
                z=z2,
                phys=phys2,
                uv=uv2,
                dist=quantile_distance(uv2, uv_target, axis_weights=axis_weights),
                cost=new_cost,
                depth=parent["depth"] + 1,
                finish_du=du,
                finish_dv=dv,
            )

            if (abs(du) <= tol[0]) and (abs(dv) <= tol[1]):
                new_finished.append(new_rec)
            else:
                expansions.append(new_rec)
        #_t4 = perf_counter()

        finished.extend(new_finished)
        if not expansions:
            #print(f"[TIMING beam] rank={_rank} step={_step} L={_L} n_cands={len(cand_meta)} uniq={len(uniq_list)} enum={_t1-_t0:.3f}s dedup={_t2-_t1:.3f}s predict={_t3-_t2:.3f}s expand={_t4-_t3:.3f}s sort=n/a", flush=True)
            break

        expansions.sort(key=lambda r: (r["dist"], r["cost"], r["depth"]))

        kept_by_end = {}
        for r in expansions:
            end = r["seq"]
            if end not in kept_by_end or (r["dist"], r["cost"], r["depth"]) < (
                kept_by_end[end]["dist"], kept_by_end[end]["cost"], kept_by_end[end]["depth"]
            ):
                kept_by_end[end] = r
        unique_expansions = list(kept_by_end.values())
        #_t5 = perf_counter()

        #print(f"[TIMING beam] rank={_rank} step={_step} L={_L} n_cands={len(cand_meta)} uniq={len(uniq_list)} enum={_t1-_t0:.3f}s dedup={_t2-_t1:.3f}s predict={_t3-_t2:.3f}s expand={_t4-_t3:.3f}s sort={_t5-_t4:.3f}s", flush=True)

        if diversity == "outcome" and len(unique_expansions) > 1 and diversity_radius > 0:
            selected = [unique_expansions[0]]
            for cand in unique_expansions[1:]:
                uv_c = cand["uv"]
                dmin = min(np.linalg.norm(uv_c - s["uv"]) for s in selected)
                if dmin >= diversity_radius:
                    selected.append(cand)
                if len(selected) >= beam_width:
                    break
            beam = selected
        else:
            beam = unique_expansions[:beam_width]

        current_best = min(r["dist"] for r in beam)
        if current_best < best_dist_so_far - min_delta:
            best_dist_so_far = current_best
            stagnant_steps = 0
        else:
            stagnant_steps += 1
        
        if patience> 0 and stagnant_steps >= patience:
            break

        if len(finished) > 5 * beam_width:
            finished.sort(key=lambda r: (r["dist"], r["cost"], r["depth"]))
            finished = finished[: 5 * beam_width]

    finished.sort(key=lambda r: (r["dist"], r["cost"], r["depth"]))
    if not finished:
        beam.sort(key=lambda r: (r["dist"], r["cost"], r["depth"]))
        return [], reconstruct_many(beam)
    return reconstruct_many(finished), reconstruct_many(beam)
