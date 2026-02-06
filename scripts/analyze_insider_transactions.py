#!/usr/bin/env python3
"""
Insider Transactions Aggregation Script

Aggregates insider transactions by:
- Position (CEO, CFO, Director, etc.)
- Year
- Transaction Type (Sale, Purchase, Gift, Grant/Exercise)

Usage:
    python scripts/analyze_insider_transactions.py AAPL
    python scripts/analyze_insider_transactions.py TSLA NVDA MSFT
    python scripts/analyze_insider_transactions.py AAPL --csv  # Save to CSV
"""

import yfinance as yf
import pandas as pd
import sys
import os
from datetime import datetime

def classify_transaction(text):
    """Classify transaction type based on text description."""
    if pd.isna(text) or text == '':
        return 'Grant/Exercise'
    text_lower = str(text).lower()
    if 'sale' in text_lower:
        return 'Sale'
    elif 'purchase' in text_lower or 'buy' in text_lower:
        return 'Purchase'
    elif 'gift' in text_lower:
        return 'Gift'
    else:
        return 'Other'


def analyze_insider_transactions(ticker: str, save_csv: bool = False, output_dir: str = None):
    """Analyze and aggregate insider transactions for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        save_csv: Whether to save results to CSV files
        output_dir: Directory to save CSV files (default: current directory)
    
    Returns:
        Dictionary with DataFrames: 'by_position', 'yearly', 'sentiment'
    """
    print(f"\n{'='*80}")
    print(f"INSIDER TRANSACTIONS ANALYSIS: {ticker.upper()}")
    print(f"{'='*80}")
    
    result = {'by_position': None, 'by_person': None, 'yearly': None, 'sentiment': None}
    
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = ticker_obj.insider_transactions
        
        if data is None or data.empty:
            print(f"No insider transaction data found for {ticker}")
            return result
        
        # Parse transaction type and year
        data['Transaction'] = data['Text'].apply(classify_transaction)
        data['Year'] = pd.to_datetime(data['Start Date']).dt.year
        
        # ============================================================
        # BY POSITION, YEAR, TRANSACTION TYPE
        # ============================================================
        print(f"\n## BY POSITION\n")
        
        agg = data.groupby(['Position', 'Year', 'Transaction']).agg({
            'Shares': 'sum',
            'Value': 'sum'
        }).reset_index()
        agg['Ticker'] = ticker.upper()
        result['by_position'] = agg
        
        for position in sorted(agg['Position'].unique()):
            print(f"\n### {position}")
            print("-" * 50)
            pos_data = agg[agg['Position'] == position].sort_values(['Year', 'Transaction'], ascending=[False, True])
            for _, row in pos_data.iterrows():
                value_str = f"${row['Value']:>15,.0f}" if pd.notna(row['Value']) and row['Value'] > 0 else f"{'N/A':>16}"
                print(f"  {row['Year']} | {row['Transaction']:15} | {row['Shares']:>12,.0f} shares | {value_str}")
        
        # ============================================================
        # BY INSIDER
        # ============================================================
        print(f"\n\n{'='*80}")
        print("INSIDER TRANSACTIONS BY PERSON")
        print(f"{'='*80}")

        insider_col = 'Insider'
        if insider_col not in data.columns and 'Name' in data.columns:
            insider_col = 'Name'
            
        if insider_col in data.columns:
            agg_person = data.groupby([insider_col, 'Position', 'Year', 'Transaction']).agg({
                'Shares': 'sum',
                'Value': 'sum'
            }).reset_index()
            agg_person['Ticker'] = ticker.upper()
            result['by_person'] = agg_person
            
            for person in sorted(agg_person[insider_col].unique()):
                print(f"\n### {str(person)}")
                print("-" * 50)
                p_data = agg_person[agg_person[insider_col] == person].sort_values(['Year', 'Transaction'], ascending=[False, True])
                for _, row in p_data.iterrows():
                    value_str = f"${row['Value']:>15,.0f}" if pd.notna(row['Value']) and row['Value'] > 0 else f"{'N/A':>16}"
                    pos_str = str(row['Position'])[:25]
                    print(f"  {row['Year']} | {pos_str:25} | {row['Transaction']:15} | {row['Shares']:>12,.0f} shares | {value_str}")
        else:
             print(f"Warning: Could not find 'Insider' or 'Name' column in data. Columns: {data.columns.tolist()}")
        
        # ============================================================
        # YEARLY SUMMARY
        # ============================================================
        print(f"\n\n{'='*80}")
        print("YEARLY SUMMARY BY TRANSACTION TYPE")
        print(f"{'='*80}")
        
        yearly = data.groupby(['Year', 'Transaction']).agg({
            'Shares': 'sum',
            'Value': 'sum'
        }).reset_index()
        yearly['Ticker'] = ticker.upper()
        result['yearly'] = yearly
        
        for year in sorted(yearly['Year'].unique(), reverse=True):
            print(f"\n{year}:")
            year_data = yearly[yearly['Year'] == year].sort_values('Transaction')
            for _, row in year_data.iterrows():
                value_str = f"${row['Value']:>15,.0f}" if pd.notna(row['Value']) and row['Value'] > 0 else f"{'N/A':>16}"
                print(f"  {row['Transaction']:15} | {row['Shares']:>12,.0f} shares | {value_str}")
        
        # ============================================================
        # OVERALL SENTIMENT
        # ============================================================
        print(f"\n\n{'='*80}")
        print("INSIDER SENTIMENT SUMMARY")
        print(f"{'='*80}\n")
        
        total_sales = data[data['Transaction'] == 'Sale']['Value'].sum()
        total_purchases = data[data['Transaction'] == 'Purchase']['Value'].sum()
        sales_count = len(data[data['Transaction'] == 'Sale'])
        purchases_count = len(data[data['Transaction'] == 'Purchase'])
        net_value = total_purchases - total_sales
        
        # Determine sentiment
        if total_purchases > total_sales:
            sentiment = "BULLISH"
        elif total_sales > total_purchases * 2:
            sentiment = "BEARISH"
        elif total_sales > total_purchases:
            sentiment = "SLIGHTLY_BEARISH"
        else:
            sentiment = "NEUTRAL"
        
        result['sentiment'] = pd.DataFrame([{
            'Ticker': ticker.upper(),
            'Total_Sales_Count': sales_count,
            'Total_Sales_Value': total_sales,
            'Total_Purchases_Count': purchases_count,
            'Total_Purchases_Value': total_purchases,
            'Net_Value': net_value,
            'Sentiment': sentiment
        }])
        
        print(f"Total Sales:      {sales_count:>5} transactions | ${total_sales:>15,.0f}")
        print(f"Total Purchases:  {purchases_count:>5} transactions | ${total_purchases:>15,.0f}")
        
        if sentiment == "BULLISH":
            print(f"\n⚡ BULLISH: Insiders are net BUYERS (${net_value:,.0f} net buying)")
        elif sentiment == "BEARISH":
            print(f"\n⚠️  BEARISH: Significant insider SELLING (${-net_value:,.0f} net selling)")
        elif sentiment == "SLIGHTLY_BEARISH":
            print(f"\n⚠️  SLIGHTLY BEARISH: More selling than buying (${-net_value:,.0f} net selling)")
        else:
            print(f"\n📊 NEUTRAL: Balanced insider activity")
        
        # Save to CSV if requested
        if save_csv:
            if output_dir is None:
                output_dir = os.getcwd()
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save by position
            by_pos_file = os.path.join(output_dir, f"insider_by_position_{ticker.upper()}_{timestamp}.csv")
            agg.to_csv(by_pos_file, index=False)
            print(f"\n📁 Saved: {by_pos_file}")

            # Save by person
            if result['by_person'] is not None:
                by_person_file = os.path.join(output_dir, f"insider_by_person_{ticker.upper()}_{timestamp}.csv")
                result['by_person'].to_csv(by_person_file, index=False)
                print(f"📁 Saved: {by_person_file}")
            
            # Save yearly summary
            yearly_file = os.path.join(output_dir, f"insider_yearly_{ticker.upper()}_{timestamp}.csv")
            yearly.to_csv(yearly_file, index=False)
            print(f"📁 Saved: {yearly_file}")
            
            # Save sentiment summary
            sentiment_file = os.path.join(output_dir, f"insider_sentiment_{ticker.upper()}_{timestamp}.csv")
            result['sentiment'].to_csv(sentiment_file, index=False)
            print(f"📁 Saved: {sentiment_file}")
            
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_insider_transactions.py TICKER [TICKER2 ...] [--csv] [--output-dir DIR]")
        print("Example: python analyze_insider_transactions.py AAPL TSLA NVDA")
        print("         python analyze_insider_transactions.py AAPL --csv")
        print("         python analyze_insider_transactions.py AAPL --csv --output-dir ./output")
        sys.exit(1)
    
    # Parse arguments
    args = sys.argv[1:]
    save_csv = '--csv' in args
    output_dir = None
    
    if '--output-dir' in args:
        idx = args.index('--output-dir')
        if idx + 1 < len(args):
            output_dir = args[idx + 1]
            args = args[:idx] + args[idx+2:]
        else:
            print("Error: --output-dir requires a directory path")
            sys.exit(1)
    
    if save_csv:
        args.remove('--csv')
    
    tickers = [t for t in args if not t.startswith('--')]
    
    # Collect all results for combined CSV
    all_by_position = []
    all_by_person = []
    all_yearly = []
    all_sentiment = []
    
    for ticker in tickers:
        result = analyze_insider_transactions(ticker, save_csv=save_csv, output_dir=output_dir)
        if result['by_position'] is not None:
            all_by_position.append(result['by_position'])
        if result['by_person'] is not None:
            all_by_person.append(result['by_person'])
        if result['yearly'] is not None:
            all_yearly.append(result['yearly'])
        if result['sentiment'] is not None:
            all_sentiment.append(result['sentiment'])
    
    # If multiple tickers and CSV mode, also save combined files
    if save_csv and len(tickers) > 1:
        if output_dir is None:
            output_dir = os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if all_by_position:
            combined_pos = pd.concat(all_by_position, ignore_index=True)
            combined_pos_file = os.path.join(output_dir, f"insider_by_position_combined_{timestamp}.csv")
            combined_pos.to_csv(combined_pos_file, index=False)
            print(f"\n📁 Combined: {combined_pos_file}")

        if all_by_person:
            combined_person = pd.concat(all_by_person, ignore_index=True)
            combined_person_file = os.path.join(output_dir, f"insider_by_person_combined_{timestamp}.csv")
            combined_person.to_csv(combined_person_file, index=False)
            print(f"📁 Combined: {combined_person_file}")
        
        if all_yearly:
            combined_yearly = pd.concat(all_yearly, ignore_index=True)
            combined_yearly_file = os.path.join(output_dir, f"insider_yearly_combined_{timestamp}.csv")
            combined_yearly.to_csv(combined_yearly_file, index=False)
            print(f"📁 Combined: {combined_yearly_file}")
        
        if all_sentiment:
            combined_sentiment = pd.concat(all_sentiment, ignore_index=True)
            combined_sentiment_file = os.path.join(output_dir, f"insider_sentiment_combined_{timestamp}.csv")
            combined_sentiment.to_csv(combined_sentiment_file, index=False)
            print(f"📁 Combined: {combined_sentiment_file}")
