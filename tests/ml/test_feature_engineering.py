import pandas as pd
import numpy as np
import pytest
from tradingagents.ml.feature_engineering import compute_features_bulk, FEATURE_COLUMNS

def _make_ohlcv(n=250):
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = pd.Series(100 + np.cumsum(rng.standard_normal(n) * 0.5), index=dates)
    return pd.DataFrame({
        "Date": dates,
        "Open": close.values * 0.999,
        "High": close.values * 1.005,
        "Low": close.values * 0.995,
        "Close": close.values,
        "Volume": rng.integers(1_000_000, 5_000_000, n),
    })

def test_feature_columns_count():
    assert len(FEATURE_COLUMNS) == 37

def test_ema_ratio_features_present():
    assert "ema14_ratio" in FEATURE_COLUMNS
    assert "ema14_slope" in FEATURE_COLUMNS

def test_regime_placeholder_columns_present():
    for col in ("spy_return_20d", "vix_level", "vix_ma20_ratio", "stock_vs_spy_20d", "sector_return_20d"):
        assert col in FEATURE_COLUMNS

def test_compute_features_bulk_returns_37_columns():
    df = _make_ohlcv()
    features = compute_features_bulk(df)
    assert set(FEATURE_COLUMNS) == set(features.columns)

def test_ema_ratio_has_values():
    df = _make_ohlcv()
    features = compute_features_bulk(df)
    valid = features["ema14_ratio"].dropna()
    assert len(valid) > 0

def test_regime_placeholders_are_nan():
    df = _make_ohlcv()
    features = compute_features_bulk(df)
    for col in ("spy_return_20d", "vix_level", "vix_ma20_ratio", "stock_vs_spy_20d", "sector_return_20d"):
        assert features[col].isna().all(), f"{col} should be all NaN (placeholder)"

def test_ema14_ratio_near_zero_for_flat_price():
    # For a constant-price series, EMA14 == close, so ratio should be ~0
    df = _make_ohlcv(n=250)
    df["Open"] = df["Close"] = df["High"] = df["Low"] = 100.0
    features = compute_features_bulk(df)
    vals = features["ema14_ratio"].dropna()
    assert (vals.abs() < 1e-10).all(), "ema14_ratio should be ~0 for flat price series"
