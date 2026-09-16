"""Feature/label normalization in the beam prediction path.

The beam and prepare_endpoints load RAW features/labels and delegate
normalization to the experts' ``predict``: ``convert_features`` (AA counts ->
fractions) + ``apply_feature_normalizer`` with each expert's persisted stats,
and label scaling via the persisted ``label_scaler1/2``. Both stages share one
path — ``predict_design`` -> ``expert.predict`` -> ``_feature_tensor`` — so it
runs identically "before the models" (prep's p_ps call) and "during the beam"
(each step). These tests lock that raw inputs are normalized with the persisted
(training-time) stats, and that ``predict`` expects raw (double-normalizing
changes the answer). Label z<->physical inversion via the persisted scalers is
covered by ``test_predict_design_phys_mean_inverts_z_mean_via_shared_scaler``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from al_pipeline.data_prep.data_loading import apply_feature_normalizer, convert_features

from tests.al_pipeline.test_moe import _build_moe_bundle, _make_raw_features_df


def test_feature_tensor_applies_convert_plus_persisted_normalizer():
    """The convert+normalize both prep and beam hit uses the expert's persisted
    stats, not a re-fit."""
    bundle = _build_moe_bundle(seed=0)
    expert = bundle.ps_expert
    raw = _make_raw_features_df(8, seed=11)

    expected = apply_feature_normalizer(
        convert_features(raw[expert.feature_columns]),
        expert.feature_normalizer_stats,
    ).to_numpy()
    got = expert._feature_tensor(raw).cpu().numpy()

    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_predict_expects_raw_not_prenormalized():
    """Feeding already-normalized features back in normalizes a second time, so
    the z-means differ from the raw path — the beam must pass raw (it does)."""
    bundle = _build_moe_bundle(seed=0)
    expert = bundle.ps_expert
    raw = _make_raw_features_df(8, seed=22)

    pred_raw = expert.predict(raw)
    prenorm = pd.DataFrame(
        expert._feature_tensor(raw).cpu().numpy(), columns=expert.feature_columns
    )
    pred_double = expert.predict(prenorm)

    assert not np.allclose(
        pred_raw["exp_density_z_mean"], pred_double["exp_density_z_mean"]
    )
