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


from tradingagents.ml.feature_engineering import SECTOR_ETF_MAP, _DEFAULT_SECTOR_ETF, YFINANCE_SECTOR_TO_ETF

def test_sector_etf_map_known_tickers():
    assert SECTOR_ETF_MAP.get("AAPL") == "XLK"
    assert SECTOR_ETF_MAP.get("JPM") == "XLF"
    assert SECTOR_ETF_MAP.get("XOM") == "XLE"

def test_sector_etf_map_default_for_unknown():
    assert _DEFAULT_SECTOR_ETF == "SPY"
    # Unknown ticker should not be in map (caller uses get() with default)
    assert SECTOR_ETF_MAP.get("UNKNOWN_TICKER", _DEFAULT_SECTOR_ETF) == "SPY"

def test_yfinance_sector_to_etf_covers_all_spdr_sectors():
    expected = {"XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLU", "XLRE", "XLB", "XLC"}
    assert set(YFINANCE_SECTOR_TO_ETF.values()) == expected

def test_get_ticker_meta_uses_static_map_first():
    from unittest.mock import patch
    from scripts.build_ml_dataset import get_ticker_meta

    # AAPL is in SECTOR_ETF_MAP → should return XLK without consulting yfinance sector string
    with patch("tradingagents.dataflows.y_finance.get_ticker_info", return_value={"marketCap": 3e12, "sector": "Consumer Electronics"}):
        cap, etf = get_ticker_meta("AAPL")
    assert etf == "XLK"
    assert cap == 3e12

def test_get_ticker_meta_falls_back_to_yfinance_sector():
    from unittest.mock import patch
    from scripts.build_ml_dataset import get_ticker_meta

    # UNKNOWN_TICKER not in SECTOR_ETF_MAP → sector resolved from yfinance
    with patch("tradingagents.dataflows.y_finance.get_ticker_info", return_value={"marketCap": 1e9, "sector": "Healthcare"}):
        cap, etf = get_ticker_meta("UNKNOWN_TICKER_XYZ")
    assert etf == "XLV"
    assert cap == 1e9

def test_get_ticker_meta_unknown_sector_uses_default():
    from unittest.mock import patch
    from scripts.build_ml_dataset import get_ticker_meta

    with patch("tradingagents.dataflows.y_finance.get_ticker_info", return_value={}):
        cap, etf = get_ticker_meta("UNKNOWN_TICKER_XYZ")
    assert etf == "SPY"
    assert cap is None

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


# --- apply_triple_barrier_labels tests ---

from tradingagents.ml.feature_engineering import apply_triple_barrier_labels

def _make_synthetic_prices(sequence):
    """Build a close price Series from a list of floats."""
    dates = pd.date_range("2023-01-01", periods=len(sequence), freq="B")
    return pd.Series(sequence, index=dates, dtype=float)


def test_binary_labels_win_case():
    # Entry at 100, upper barrier at 105 (+5%), lower at 97 (-3%), 20d window
    # Day 5 price = 106 → hits upper → WIN=1
    prices = [100.0] + [101.0] * 4 + [106.0] + [100.0] * 20
    close = _make_synthetic_prices(prices)
    labels = apply_triple_barrier_labels(close, profit_target=0.05, stop_loss=0.03,
                                          max_holding_days=20, binary=True)
    assert labels.iloc[0] == 1.0, "Should be WIN (1)"


def test_binary_labels_loss_mapped_to_zero():
    # Day 3 price = 96 → hits lower barrier (-4%) → LOSS → in binary mode = 0
    prices = [100.0, 100.0, 100.0, 96.0] + [100.0] * 20
    close = _make_synthetic_prices(prices)
    labels = apply_triple_barrier_labels(close, profit_target=0.05, stop_loss=0.03,
                                          max_holding_days=20, binary=True)
    assert labels.iloc[0] == 0.0, "LOSS should map to NOT-WIN (0) in binary mode"


def test_binary_labels_timeout_is_zero():
    # Price drifts +2% but never hits +5% or -3% in 5 days → TIMEOUT → 0
    prices = [100.0, 101.0, 101.5, 102.0, 102.5, 102.0] + [100.0] * 5
    close = _make_synthetic_prices(prices)
    labels = apply_triple_barrier_labels(close, profit_target=0.05, stop_loss=0.03,
                                          max_holding_days=5, binary=True)
    assert labels.iloc[0] == 0.0, "Partial gain (timeout) should be NOT-WIN (0)"


def test_triclass_labels_loss_is_minus_one():
    # Same setup as loss test but binary=False → should be -1
    prices = [100.0, 100.0, 100.0, 96.0] + [100.0] * 20
    close = _make_synthetic_prices(prices)
    labels = apply_triple_barrier_labels(close, profit_target=0.05, stop_loss=0.03,
                                          max_holding_days=20, binary=False)
    assert labels.iloc[0] == -1.0, "LOSS should be -1 in 3-class mode"


def test_binary_only_zero_and_one():
    import numpy as np
    rng = np.random.default_rng(99)
    prices = 100 + np.cumsum(rng.standard_normal(150) * 0.5)
    close = _make_synthetic_prices(list(prices))
    labels = apply_triple_barrier_labels(close, profit_target=0.05, stop_loss=0.03,
                                          max_holding_days=20, binary=True)
    unique = set(labels.dropna().unique())
    assert unique.issubset({0.0, 1.0}), f"Binary labels should only be 0/1, got {unique}"
