from pathlib import Path

from phase_runner.feature_store import FeatureSpec, FeatureStore


def test_feature_store_window_features():
    store = FeatureStore(FeatureSpec(leakage_check_sample=5))
    claims = store.load_claims(Path("data/synthetic_claims.csv"))
    features, target = store.build_features(claims)

    assert len(features) == len(target)
    assert features.isnull().sum().sum() == 0
    # Ensure window features were created
    expected_cols = {"payer_30d_denial_rate", "payer_cpt_dx_60d_denial_rate"}
    assert expected_cols.issubset(set(features.columns))
