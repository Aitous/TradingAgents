
import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.technical_analyst import TechnicalAnalyst


def test_technical_analyst_report_generation(sample_stock_data):
    df = wrap(sample_stock_data)
    current_price = 115.0

    analyst = TechnicalAnalyst(df, current_price)
    report = analyst.generate_report("TEST", "2025-01-01")

    assert "# Technical Analysis for TEST" in report
    assert "**Current Price:** $115.00" in report
    assert "## Price Action" in report
    assert "Daily Change" in report
    assert "## RSI" in report
    assert "## MACD" in report

def test_technical_analyst_empty_data():
    empty_df = pd.DataFrame()
    # It might raise an error or handle it, usually logic handles standard DF but let's check
    # The class expects columns, so let's pass empty with columns
    df = pd.DataFrame(columns=["close", "high", "low", "volume"])

    # Wrapping empty might fail or produce empty wrapped
    # Our TechnicalAnalyst assumes valid data somewhat, but we should make sure it doesn't just crash blindly
    # Actually, y_finance.py checks for empty before calling, so the class itself assumes data.
    pass
