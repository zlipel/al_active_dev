"""Beam-search kernel — Row 9 refactor.

The kernel no longer knows about surrogates, quantile transforms, or label
scalers. Everything policy-specific lives on ``policy`` (a `BeamPolicy`);
the kernel just calls ``policy.predict_candidates(seqs)`` at every step.
This lets Row 10 stack analysis + policy comparisons on top of the same
kernel without touching this file again.

Node record shape (extended for Row 9):

  seq, parent, edit, z, phys, uv, dist, cost, depth, finish_du, finish_dv,
  p_ps, reason

``p_ps`` is the per-candidate gate probability under the current surrogate
(``None`` for global). ``reason`` is one of ``"ok"``, ``"invalid_phys"``,
``"rejected_by_gate"``; ``ok`` for finished paths, other values only appear
on invalid candidates that never became a node (they're filtered before
``make_node``).
"""
from __future__ import annotations

import numpy as np
from mpi4py import MPI

from beam_search.policy import BeamPolicy


AA = list("ACDEFGHIKLMNPQRSTVWY")


def edit_unit_cost(op: str) -> int:
    return 1 if op == 'sub' else 2


def apply_edit(seq, op, pos, aa=None):
    if op == 'sub':
        return seq[:pos] + aa + seq[pos + 1:]
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

def predict_candidate_frames(policy: BeamPolicy, seq_list):
    """Thin adapter around ``policy.predict_candidates`` for the beam loop.

    Preserves the pre-Row-9 return-dict shape ({z, phys, uv, valid}) so the
    kernel body reads the same, and adds two new keys the kernel uses to
    populate the node record: ``p_ps`` and ``reason``.
    """
    pred = policy.predict_candidates(seq_list)
    return {
        "z":      pred.z_mean,
        "phys":   pred.phys,
        "uv":     pred.uv,
        "valid":  pred.valid,
        "p_ps":   pred.p_ps,
        "reason": pred.reason,
    }


def quantile_distance(uv, uv_target, axis_weights=(1.0, 1.0)):
    du = (uv[0] - uv_target[0]) / axis_weights[0]
    dv = (uv[1] - uv_target[1]) / axis_weights[1]
    return float(np.sqrt(du * du + dv * dv))


# ---------------- lightweight node + reconstruction ----------------

def make_node(seq, parent, edit, z, phys, uv, dist, cost, depth,
              finish_du, finish_dv, p_ps=None, reason="ok"):
    return {
        "seq":        seq,
        "parent":     parent,
        "edit":       edit,
        "z":          z,
        "phys":       phys,
        "uv":         uv,
        "dist":       dist,
        "cost":       cost,
        "depth":      depth,
        "finish_du":  finish_du,
        "finish_dv":  finish_dv,
        "p_ps":       p_ps,
        "reason":     reason,
    }


def reconstruct_record(node):
    path = []
    edits = []
    preds_z = []
    preds_phys = []
    preds_uv = []
    preds_p_ps = []
    reasons = []

    cur = node
    while cur is not None:
        path.append(cur["seq"])
        preds_z.append(cur["z"])
        preds_phys.append(cur["phys"])
        preds_uv.append(cur["uv"])
        preds_p_ps.append(cur.get("p_ps", np.nan))
        reasons.append(cur.get("reason", "ok"))
        if cur["edit"] is not None:
            edits.append(cur["edit"])
        cur = cur["parent"]

    path.reverse()
    edits.reverse()
    preds_z.reverse()
    preds_phys.reverse()
    preds_uv.reverse()
    preds_p_ps.reverse()
    reasons.reverse()

    return {
        "path":       path,
        "edits":      edits,
        "preds_z":    np.vstack(preds_z),
        "preds_phys": np.vstack(preds_phys),
        "preds_uv":   np.vstack(preds_uv),
        "preds_p_ps": np.array([np.nan if v is None else v for v in preds_p_ps], dtype=float),
        "reasons":    list(reasons),
        "dist":       node["dist"],
        "cost":       node["cost"],
        "finish_du":  node["finish_du"],
        "finish_dv":  node["finish_dv"],
        # Endpoint drift diagnostic per §IV.endpoints — record the tip's
        # gate probability so the endpoints CSV can flag "policy is wrong"
        # vs. "beam walked out of the chosen expert's domain".
        "endpoint_p_ps": (float(node["p_ps"]) if node.get("p_ps") is not None else float("nan")),
    }


def reconstruct_many(nodes):
    return [reconstruct_record(n) for n in nodes]


# ---------------- search ----------------

def beam_search_paths(
    policy: BeamPolicy,
    start_seq: str,
    uv_target,
    start_phys,
    start_uv,
    start_z=None,
    beam_width: int = 32,
    max_steps: int = 8,
    tol=(0.02, 0.02),
    diversity: str = "none",
    diversity_radius: float = 0.0,
    budget_cost=None,
    length_changes: bool = True,
    axis_weights=(1.0, 1.0),
    min_positive: float = 1e-12,
    patience: int = 0,
    min_delta: float = 0.0,
):
    """Run beam search from ``start_seq`` toward ``uv_target``.

    The policy owns the surrogate, featurizer, quantile transforms, and any
    gate thresholds — the kernel only calls ``policy.predict_candidates``.
    ``min_positive`` here is a redundancy safety net; policy already applies
    the same physical-validity check with the same default. Kept as a kernel
    kwarg so callers can still tighten it independently of the policy.
    """
    uv_target = np.asarray(uv_target, dtype=float)
    phys0 = np.asarray(start_phys, dtype=float)
    uv0 = np.asarray(start_uv, dtype=float)

    if start_z is None:
        s_rho, s_diff = policy.label_scalers
        z0 = np.array([
            float(s_rho.transform([[phys0[0]]])[0, 0]),
            float(s_diff.transform([[phys0[1]]])[0, 0]),
        ], dtype=float)
    else:
        z0 = np.asarray(start_z, dtype=float)

    if not (np.isfinite(phys0[0]) and np.isfinite(phys0[1])
            and phys0[0] > min_positive and phys0[1] > min_positive):
        raise ValueError(f"Start sequence {start_seq!r} has invalid provided start_phys={phys0}.")

    # Best-effort start p_ps for the drift diagnostic. Cheap: a single-row
    # predict on the start's featurized row. Uses the policy's own surrogate
    # so the p_ps here matches what candidates get.
    start_pred = policy.predict_candidates([start_seq])
    start_p_ps = float(start_pred.p_ps[0]) if start_pred.p_ps is not None else None

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
        p_ps=start_p_ps,
        reason="ok",
    )

    finished = []
    beam = [start]
    _rank = MPI.COMM_WORLD.Get_rank()
    best_dist_so_far = np.inf
    stagnant_steps = 0

    for _step in range(max_steps):
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

        if not cand_meta:
            break

        uniq_map = {}
        uniq_list = []
        for _, _, _, _, s2, _ in cand_meta:
            if s2 not in uniq_map:
                uniq_map[s2] = len(uniq_list)
                uniq_list.append(s2)

        pred = predict_candidate_frames(policy, uniq_list)

        expansions = []
        new_finished = []
        for (bi, op, pos, aa, s2, new_cost) in cand_meta:
            k = uniq_map[s2]
            if not pred["valid"][k]:
                continue

            z2 = pred["z"][k]
            phys2 = pred["phys"][k]
            uv2 = pred["uv"][k]
            p_ps2 = (float(pred["p_ps"][k]) if pred["p_ps"] is not None else None)
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
                p_ps=p_ps2,
                reason="ok",
            )

            if (abs(du) <= tol[0]) and (abs(dv) <= tol[1]):
                new_finished.append(new_rec)
            else:
                expansions.append(new_rec)

        finished.extend(new_finished)
        if not expansions:
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

        if patience > 0 and stagnant_steps >= patience:
            break

        if len(finished) > 5 * beam_width:
            finished.sort(key=lambda r: (r["dist"], r["cost"], r["depth"]))
            finished = finished[: 5 * beam_width]

    finished.sort(key=lambda r: (r["dist"], r["cost"], r["depth"]))
    if not finished:
        beam.sort(key=lambda r: (r["dist"], r["cost"], r["depth"]))
        return [], reconstruct_many(beam)
    return reconstruct_many(finished), reconstruct_many(beam)
