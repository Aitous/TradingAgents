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


from tradingagents.ml.feature_engineering import SECTOR_ETF_MAP, _DEFAULT_SECTOR_ETF

def test_sector_etf_map_known_tickers():
    assert SECTOR_ETF_MAP.get("AAPL") == "XLK"
    assert SECTOR_ETF_MAP.get("JPM") == "XLF"
    assert SECTOR_ETF_MAP.get("XOM") == "XLE"

def test_sector_etf_map_default_for_unknown():
    assert _DEFAULT_SECTOR_ETF == "SPY"
    # Unknown ticker should not be in map (caller uses get() with default)
    assert SECTOR_ETF_MAP.get("UNKNOWN_TICKER", _DEFAULT_SECTOR_ETF) == "SPY"

def test_fetch_market_context_returns_correct_columns():
    # Test the return structure without network (mock yfinance)
    from unittest.mock import patch
    import pandas as pd
    import numpy as np
    from tradingagents.ml.feature_engineering import fetch_market_context

    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    spy = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5), index=dates)
    vix = pd.Series(15 + np.random.randn(50), index=dates)

    # Build MultiIndex DataFrame like yfinance returns
    cols = pd.MultiIndex.from_tuples([("Close", "SPY"), ("Close", "^VIX")])
    mock_df = pd.DataFrame(
        np.column_stack([spy.values, vix.values]),
        index=dates,
        columns=cols,
    )

    with patch("yfinance.download", return_value=mock_df):
        ctx = fetch_market_context("2024-01-01", "2024-03-01")

    assert "spy_return_20d" in ctx.columns
    assert "vix_level" in ctx.columns
    assert "vix_ma20_ratio" in ctx.columns


def test_fetch_market_context_flat_column_fallback():
    # Simulates yfinance returning flat columns (single-ticker collapse edge case)
    from unittest.mock import patch, call
    import pandas as pd
    import numpy as np
    from tradingagents.ml.feature_engineering import fetch_market_context

    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    spy_vals = 100 + np.cumsum(np.random.default_rng(1).standard_normal(50) * 0.5)
    vix_vals = 15 + np.random.default_rng(2).standard_normal(50)

    # First call returns flat columns (triggers fallback)
    flat_df = pd.DataFrame({"Close": spy_vals, "Open": spy_vals}, index=dates)
    spy_df = pd.DataFrame({"Close": spy_vals}, index=dates)
    vix_df = pd.DataFrame({"Close": vix_vals}, index=dates)

    def side_effect(*args, **kwargs):
        tickers = args[0] if args else kwargs.get("tickers", [])
        if isinstance(tickers, list):
            return flat_df  # batch call → flat (triggers fallback)
        if tickers == "SPY":
            return spy_df
        return vix_df

    with patch("yfinance.download", side_effect=side_effect):
        ctx = fetch_market_context("2024-01-01", "2024-03-01")

    assert "spy_return_20d" in ctx.columns
    assert "vix_level" in ctx.columns
    assert len(ctx) > 0
