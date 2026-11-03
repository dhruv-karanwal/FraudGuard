import numpy as np
import pandas as pd
import pytest

from src.features import TAB_FEATURES, FeatureScaler, raw_features


def _df():
    return pd.DataFrame({
        "step": [1, 13, 25],
        "type": pd.Categorical(["TRANSFER", "CASH_OUT", "PAYMENT"]),
        "amount": [100.0, 50.0, 10.0],
        "oldbalanceOrg": [100.0, 0.0, 30.0],
        "newbalanceOrig": [0.0, 0.0, 20.0],
        "oldbalanceDest": [0.0, 10.0, 0.0],
        "newbalanceDest": [0.0, 60.0, 0.0],
        "isFraud": [1, 0, 0],
        "isFlaggedFraud": [1, 0, 0],
    })


def test_raw_features_values():
    f = raw_features(_df())
    assert f.loc[0, "err_balance_orig"] == 0.0            # 0 + 100 - 100
    assert f.loc[0, "err_balance_dest"] == pytest.approx(np.log1p(100))
    assert f.loc[1, "err_balance_orig"] == pytest.approx(np.log1p(50))
    assert f.loc[1, "err_balance_dest"] == 0.0            # 10 + 50 - 60
    assert f.loc[0, "orig_emptied"] == 1 and f.loc[1, "orig_emptied"] == 0
    assert f.loc[1, "orig_zero_before"] == 1
    assert f.loc[0, "hour_sin"] == pytest.approx(f.loc[2, "hour_sin"])  # step 1 and 25
    assert f[["type_TRANSFER", "type_CASH_OUT", "type_PAYMENT"]].to_numpy().diagonal().tolist() == [1, 1, 1]


def test_no_label_or_leaky_column_in_features():
    f = raw_features(_df())
    for col in ("isFraud", "isFlaggedFraud"):
        assert col not in f.columns
    assert set(TAB_FEATURES) <= set(f.columns)


def test_scaler_uses_train_statistics_only():
    train = raw_features(_df().iloc[:2])
    test = raw_features(_df().iloc[2:])
    sc = FeatureScaler(TAB_FEATURES).fit(train)
    i = TAB_FEATURES.index("log_amount")
    mu = train["log_amount"].mean()
    sd = train["log_amount"].std(ddof=0)
    out = sc.transform(test)
    assert out[0, i] == pytest.approx((test["log_amount"].iloc[0] - mu) / sd, rel=1e-5)
    # Binary columns pass through unscaled.
    j = TAB_FEATURES.index("type_TRANSFER")
    assert sc.transform(train)[:, j].tolist() == [1.0, 0.0]


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        FeatureScaler(TAB_FEATURES).transform(raw_features(_df()))
