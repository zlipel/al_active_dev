import pandas as pd
import numpy as np


def dominates(sol1, sol2, kind=('max','max'), eps=1e-12):
    """Return True if sol1 Pareto-dominates sol2 with numerical tolerance eps."""
    a1, a2 = sol1[0], sol1[1]
    b1, b2 = sol2[0], sol2[1]
    s1, s2 = kind

    # helper: compare with tolerance
    def ge(x, y): return x >= y - eps
    def gt(x, y): return x >  y + eps
    def le(x, y): return x <= y + eps
    def lt(x, y): return x <  y - eps

    if s1 == 'max' and s2 == 'max':
        return ge(a1,b1) and ge(a2,b2) and (gt(a1,b1) or gt(a2,b2))
    if s1 == 'max' and s2 == 'min':
        return ge(a1,b1) and le(a2,b2) and (gt(a1,b1) or lt(a2,b2))
    if s1 == 'min' and s2 == 'max':
        return le(a1,b1) and ge(a2,b2) and (lt(a1,b1) or gt(a2,b2))
    if s1 == 'min' and s2 == 'min':
        return le(a1,b1) and le(a2,b2) and (lt(a1,b1) or lt(a2,b2))
    raise ValueError("kind must be a pair from {'max','min'}.")

def find_pareto_front(labels: pd.DataFrame,
                      kind=('max','max'),
                      objectives=('exp_density','diff'),
                      eps=1e-12):
    """
    Return nondominated dataframe and original indices.
    kind=('max','max') for upper front; ('max','min') etc. for mixed.
    """
    data = labels[list(objectives)].copy()
    data['__idx__'] = labels.index
    data = data.dropna(subset=objectives).reset_index(drop=True)

    vals = data[list(objectives)].to_numpy()
    n = vals.shape[0]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[i]:
                continue
            if dominates(vals[j], vals[i], kind=kind, eps=eps):
                keep[i] = False

    nd_df = data.loc[keep, list(objectives)].reset_index(drop=True)
    nd_indices = data.loc[keep, '__idx__'].tolist()
    return nd_df, nd_indices
