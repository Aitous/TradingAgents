"""
Historical Memory Builder for TradingAgents

This module creates agent memories from historical stock data by:
1. Finding high movers (>15% in 5 days)
2. Running retrospective trading graph analysis at T-7 and T-30 days before the move
3. Extracting structured signals and agent decisions
4. Creating situation -> outcome mappings with enhanced metadata
5. Storing memories in ChromaDB for future retrieval
"""

import os
import re
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from tradingagents.tools.executor import execute_tool
from tradingagents.agents.utils.memory import FinancialSituationMemory


class HistoricalMemoryBuilder:
    """Build agent memories from historical stock data."""

    def __init__(self, config: dict):
        """Initialize the memory builder.

        Args:
            config: TradingAgents configuration dictionary
        """
        self.config = config
        self.memories_created = {
            "bull": 0,
            "bear": 0,
            "trader": 0,
            "invest_judge": 0,
            "risk_manager": 0
        }

    def get_tickers_from_alpha_vantage(self, limit: int = 20) -> List[str]:
        """
        Get ticker list from Alpha Vantage top gainers/losers.

        Args:
            limit: Number of tickers to get from each category (gainers/losers)

        Returns:
            List of ticker symbols from top gainers and losers
        """
        print(f"\n🔍 Fetching top movers from Alpha Vantage...")

        try:
            # Use execute_tool to call the alpha vantage function
            response = execute_tool("get_market_movers", limit=limit)

            # Parse the markdown table response to extract tickers
            tickers = set()

            lines = response.split('\n')
            for line in lines:
                # Look for table rows with ticker data
                if '|' in line and not line.strip().startswith('|---'):
                    parts = [p.strip() for p in line.split('|')]
                    # Table format: | Ticker | Price | Change % | Volume |
                    if len(parts) >= 2 and parts[1] and parts[1] not in ['Ticker', '']:
                        ticker = parts[1].strip()

                        # Filter out warrants, units, and problematic tickers
                        if ticker and self._is_valid_ticker(ticker):
                            tickers.add(ticker)

            ticker_list = sorted(list(tickers))
            print(f"   ✅ Found {len(ticker_list)} unique tickers from Alpha Vantage")
            print(f"   Tickers: {', '.join(ticker_list[:10])}{'...' if len(ticker_list) > 10 else ''}")

            return ticker_list

        except Exception as e:
            print(f"   ⚠️  Error fetching from Alpha Vantage: {e}")
            print(f"   Falling back to empty list")
            return []

    def _is_valid_ticker(self, ticker: str) -> bool:
        """
        Validate if a ticker is suitable for analysis.

        Filters out:
        - Warrants (ending in W, WW, WS)
        - Units (ending in U)
        - Preferred shares (containing -, /)
        - Rights (ending in R)
        - Other derivative instruments

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if ticker is a regular stock, False otherwise
        """
        if not ticker or len(ticker) > 6:
            return False

        # Must be uppercase letters and numbers only
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            return False

        # Filter out warrants (W, WW, WS suffix)
        if ticker.endswith('W') or ticker.endswith('WW') or ticker.endswith('WS'):
            return False

        # Filter out units
        if ticker.endswith('U'):
            return False

        # Filter out rights
        if ticker.endswith('R') and len(ticker) > 1:
            return False

        # Filter out other suffixes that indicate derivatives
        if ticker.endswith('Z'):  # Often used for special situations
            return False

        return True

    def find_high_movers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        min_move_pct: float = 15.0,
        window_days: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find stocks that had significant moves (>15% in 5 days).

        Args:
            tickers: List of tickers to scan
            start_date: Start date for scanning (YYYY-MM-DD)
            end_date: End date for scanning (YYYY-MM-DD)
            min_move_pct: Minimum percentage move (default: 15%)
            window_days: Rolling window in days (default: 5)

        Returns:
            List of dicts with keys:
                - ticker: Stock symbol
                - move_start_date: Start of the move (YYYY-MM-DD)
                - move_end_date: End of the move (YYYY-MM-DD)
                - move_pct: Percentage change
                - direction: "up" or "down"
                - start_price: Price at start
                - end_price: Price at end
        """
        high_movers = []

        print(f"\n🔍 Scanning for high movers ({min_move_pct}%+ in {window_days} days)")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Tickers: {len(tickers)}\n")

        for ticker in tickers:
            try:
                print(f"   Scanning {ticker}...", end=" ")

                # Download historical data using yfinance
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)

                if df.empty:
                    print("No data")
                    continue

                # Calculate rolling returns over window_days
                df['rolling_return'] = df['Close'].pct_change(periods=window_days) * 100

                # Find periods with moves >= min_move_pct
                significant_moves = df[abs(df['rolling_return']) >= min_move_pct]

                if not significant_moves.empty:
                    for idx, row in significant_moves.iterrows():
                        # Get the start date (window_days before this date)
                        move_end_date = idx.strftime('%Y-%m-%d')
                        move_start_date = (idx - timedelta(days=window_days)).strftime('%Y-%m-%d')

                        # Get prices
                        try:
                            start_price = df.loc[df.index >= move_start_date, 'Close'].iloc[0]
                            end_price = row['Close']
                            move_pct = row['rolling_return']

                            high_movers.append({
                                'ticker': ticker,
                                'move_start_date': move_start_date,
                                'move_end_date': move_end_date,
                                'move_pct': move_pct,
                                'direction': 'up' if move_pct > 0 else 'down',
                                'start_price': start_price,
                                'end_price': end_price
                            })
                        except (IndexError, KeyError):
                            continue

                    print(f"Found {len([m for m in high_movers if m['ticker'] == ticker])} moves")
                else:
                    print("No significant moves")

            except Exception as e:
                print(f"Error: {e}")
                continue

        print(f"\n✅ Total high movers found: {len(high_movers)}\n")
        return high_movers

    def run_retrospective_analysis(
        self,
        ticker: str,
        analysis_date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Run the trading graph analysis for a ticker at a specific historical date.

        This simulates what the agent would have seen/decided on that date.

        Args:
            ticker: Stock ticker symbol
            analysis_date: Date to run analysis (YYYY-MM-DD)

        Returns:
            Dict with keys:
                - market_report: str
                - sentiment_report: str
                - news_report: str
                - fundamentals_report: str
                - investment_plan: str (if available)
                - final_decision: str (if available)
                - structured_signals: Dict of extracted features
        """
        try:
            # Import here to avoid circular imports
            from tradingagents.graph.trading_graph import TradingAgentsGraph

            print(f"      Running analysis for {ticker} on {analysis_date}...")

            # Create trading graph instance
            # Use fewer analysts to reduce token usage
            graph = TradingAgentsGraph(
                selected_analysts=["market", "fundamentals"],  # Skip social/news to reduce tokens
                config=self.config,
                debug=False
            )

            # Run the analysis (returns tuple: final_state, processed_signal)
            final_state, _ = graph.propagate(ticker, analysis_date)

            # Extract reports and decisions (with type safety)
            def safe_get_str(d, key, default=''):
                """Safely extract string from state, handling lists or other types."""
                value = d.get(key, default)
                if isinstance(value, list):
                    # If it's a list, try to extract text from messages
                    return ' '.join(str(item) for item in value)
                return str(value) if value else default

            # Extract reports and decisions
            analysis_data = {
                'market_report': safe_get_str(final_state, 'market_report'),
                'sentiment_report': safe_get_str(final_state, 'sentiment_report'),
                'news_report': safe_get_str(final_state, 'news_report'),
                'fundamentals_report': safe_get_str(final_state, 'fundamentals_report'),
                'investment_plan': safe_get_str(final_state, 'investment_plan'),
                'final_decision': safe_get_str(final_state, 'final_trade_decision'),
            }

            # Extract structured signals from reports
            analysis_data['structured_signals'] = self.extract_structured_signals(analysis_data)

            return analysis_data

        except Exception as e:
            print(f"      Error running analysis: {e}")
            import traceback
            print(f"      Traceback: {traceback.format_exc()}")
            return None

    def extract_structured_signals(self, reports: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract structured signal features from analyst reports.

        Args:
            reports: Dict with market_report, sentiment_report, news_report, fundamentals_report

        Returns:
            Dict with extracted signal features:
                - unusual_volume: bool
                - analyst_sentiment: str (bullish/bearish/neutral)
                - news_sentiment: str (positive/negative/neutral)
                - short_interest: str (high/medium/low)
                - insider_activity: str (buying/selling/none)
                - price_trend: str (uptrend/downtrend/sideways)
                - volatility: str (high/medium/low)
        """
        signals = {}

        market_report = reports.get('market_report', '')
        sentiment_report = reports.get('sentiment_report', '')
        news_report = reports.get('news_report', '')
        fundamentals_report = reports.get('fundamentals_report', '')

        # Extract volume signals
        signals['unusual_volume'] = bool(
            re.search(r'(unusual volume|volume spike|high volume|increased volume)', market_report, re.IGNORECASE)
        )

        # Extract sentiment
        if re.search(r'(bullish|positive outlook|strong buy|buy)', sentiment_report + news_report, re.IGNORECASE):
            signals['analyst_sentiment'] = 'bullish'
        elif re.search(r'(bearish|negative outlook|strong sell|sell)', sentiment_report + news_report, re.IGNORECASE):
            signals['analyst_sentiment'] = 'bearish'
        else:
            signals['analyst_sentiment'] = 'neutral'

        # Extract news sentiment
        if re.search(r'(positive|good news|beat expectations|upgrade|growth)', news_report, re.IGNORECASE):
            signals['news_sentiment'] = 'positive'
        elif re.search(r'(negative|bad news|miss expectations|downgrade|decline)', news_report, re.IGNORECASE):
            signals['news_sentiment'] = 'negative'
        else:
            signals['news_sentiment'] = 'neutral'

        # Extract short interest
        if re.search(r'(high short interest|heavily shorted|short squeeze)', market_report + news_report, re.IGNORECASE):
            signals['short_interest'] = 'high'
        elif re.search(r'(low short interest|minimal short)', market_report, re.IGNORECASE):
            signals['short_interest'] = 'low'
        else:
            signals['short_interest'] = 'medium'

        # Extract insider activity
        if re.search(r'(insider buying|executive purchased|insider purchases)', news_report + fundamentals_report, re.IGNORECASE):
            signals['insider_activity'] = 'buying'
        elif re.search(r'(insider selling|executive sold|insider sales)', news_report + fundamentals_report, re.IGNORECASE):
            signals['insider_activity'] = 'selling'
        else:
            signals['insider_activity'] = 'none'

        # Extract price trend
        if re.search(r'(uptrend|bullish trend|rising|moving higher|higher highs)', market_report, re.IGNORECASE):
            signals['price_trend'] = 'uptrend'
        elif re.search(r'(downtrend|bearish trend|falling|moving lower|lower lows)', market_report, re.IGNORECASE):
            signals['price_trend'] = 'downtrend'
        else:
            signals['price_trend'] = 'sideways'

        # Extract volatility
        if re.search(r'(high volatility|volatile|wild swings|sharp movements)', market_report, re.IGNORECASE):
            signals['volatility'] = 'high'
        elif re.search(r'(low volatility|stable|steady)', market_report, re.IGNORECASE):
            signals['volatility'] = 'low'
        else:
            signals['volatility'] = 'medium'

        return signals

    def build_memories_from_high_movers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        min_move_pct: float = 15.0,
        analysis_windows: List[int] = [7, 30],
        max_samples: int = 50,
        sample_strategy: str = "diverse"
    ) -> Dict[str, FinancialSituationMemory]:
        """
        Build memories by finding high movers and running retrospective analyses.

        This is the main method for the new learning system.

        Args:
            tickers: List of tickers to scan
            start_date: Start date for scanning (YYYY-MM-DD)
            end_date: End date for scanning (YYYY-MM-DD)
            min_move_pct: Minimum percentage move (default: 15%)
            analysis_windows: Days before move to analyze (default: [7, 30])
            max_samples: Maximum number of high movers to analyze (default: 50)
            sample_strategy: How to select samples from high movers:
                - "diverse": Mix of up/down moves, different magnitudes (recommended)
                - "largest": Take the largest moves only
                - "recent": Take the most recent moves only
                - "random": Random sampling

        Returns:
            Dictionary of populated memory instances for each agent type
        """
        print("=" * 70)
        print("🏗️  BUILDING MEMORIES FROM HIGH MOVERS")
        print("=" * 70)

        # Step 1: Find high movers
        high_movers = self.find_high_movers(tickers, start_date, end_date, min_move_pct)

        if not high_movers:
            print("⚠️  No high movers found. Try a different date range or lower threshold.")
            return {}

        # Step 1.5: Sample/filter high movers based on strategy
        sampled_movers = self._sample_high_movers(high_movers, max_samples, sample_strategy)

        print(f"\n📊 Sampling Strategy: {sample_strategy}")
        print(f"   Total high movers found: {len(high_movers)}")
        print(f"   Samples to analyze: {len(sampled_movers)}")
        print(f"   Estimated runtime: ~{len(sampled_movers) * len(analysis_windows) * 2} minutes")
        print()

        # Initialize memory stores
        agent_memories = {
            "bull": FinancialSituationMemory("bull_memory", self.config),
            "bear": FinancialSituationMemory("bear_memory", self.config),
            "trader": FinancialSituationMemory("trader_memory", self.config),
            "invest_judge": FinancialSituationMemory("invest_judge_memory", self.config),
            "risk_manager": FinancialSituationMemory("risk_manager_memory", self.config)
        }

        # Step 2: For each high mover, run retrospective analyses
        print("\n📊 Running retrospective analyses...\n")

        for idx, mover in enumerate(sampled_movers, 1):
            ticker = mover['ticker']
            move_pct = mover['move_pct']
            direction = mover['direction']
            move_start_date = mover['move_start_date']

            print(f"   [{idx}/{len(sampled_movers)}] {ticker}: {move_pct:+.1f}% {direction}")

            # Run analyses at different time windows before the move
            for days_before in analysis_windows:
                # Calculate analysis date
                try:
                    analysis_date = (
                        datetime.strptime(move_start_date, '%Y-%m-%d') - timedelta(days=days_before)
                    ).strftime('%Y-%m-%d')

                    print(f"      Analyzing T-{days_before} days ({analysis_date})...")

                    # Run trading graph analysis
                    analysis = self.run_retrospective_analysis(ticker, analysis_date)

                    if not analysis:
                        print(f"      ⚠️  Analysis failed, skipping...")
                        continue

                    # Create combined situation text
                    situation_text = f"""
**Ticker**: {ticker}
**Analysis Date**: {analysis_date}
**Time Before Move**: {days_before} days

**Market Analysis**:
{analysis['market_report'][:500]}...

**Sentiment Analysis**:
{analysis['sentiment_report'][:500]}...

**News Analysis**:
{analysis['news_report'][:500]}...

**Fundamentals**:
{analysis['fundamentals_report'][:500]}...
""".strip()

                    # Extract agent recommendation from investment plan and final decision
                    agent_recommendation = self._extract_recommendation(
                        analysis.get('investment_plan', ''),
                        analysis.get('final_decision', '')
                    )

                    # Determine if agent was correct
                    was_correct = self._compute_correctness(agent_recommendation, direction)

                    # Create metadata
                    metadata = {
                        'ticker': ticker,
                        'analysis_date': analysis_date,
                        'days_before_move': days_before,
                        'move_pct': abs(move_pct),
                        'move_direction': direction,
                        'agent_recommendation': agent_recommendation,
                        'was_correct': was_correct,
                        'structured_signals': analysis['structured_signals']
                    }

                    # Create recommendation text
                    lesson_text = f"This signal combination is reliable for predicting {direction} moves." if was_correct else "This signal combination can be misleading. Need to consider other factors."

                    recommendation_text = f"""
Agent Decision: {agent_recommendation}
Actual Outcome: {direction} {abs(move_pct):.1f}%
Correctness: {'✓ CORRECT' if was_correct else '✗ INCORRECT'}

{days_before} days before this {direction} move, the agent recommended {agent_recommendation}.
The stock moved {direction} by {abs(move_pct):.1f}%, so the agent was {'correct' if was_correct else 'incorrect'}.

Structured Signals Present:
{self._format_signals(analysis.get('structured_signals', {}))}

Lesson: {lesson_text}
""".strip()

                    # Store in all agent memories
                    for agent_type, memory in agent_memories.items():
                        memory.add_situations_with_metadata([
                            (situation_text, recommendation_text, metadata)
                        ])

                    self.memories_created[agent_type] = self.memories_created.get(agent_type, 0) + 1

                    print(f"      ✅ Memory created: {agent_recommendation} -> {direction} ({was_correct})")

                except Exception as e:
                    print(f"      ⚠️  Error: {e}")
                    continue

        # Print summary
        print("\n" + "=" * 70)
        print("📊 MEMORY CREATION SUMMARY")
        print("=" * 70)
        print(f"   High movers analyzed: {len(sampled_movers)}")
        print(f"   Analysis windows: {analysis_windows} days before move")
        for agent_type, count in self.memories_created.items():
            print(f"   {agent_type.ljust(15)}: {count} memories")

        # Print statistics
        print("\n📈 MEMORY BANK STATISTICS")
        print("=" * 70)
        for agent_type, memory in agent_memories.items():
            stats = memory.get_statistics()
            print(f"\n   {agent_type.upper()}:")
            print(f"      Total memories: {stats['total_memories']}")
            print(f"      Accuracy rate: {stats['accuracy_rate']:.1f}%")
            print(f"      Avg move: {stats['avg_move_pct']:.1f}%")

        print("=" * 70 + "\n")

        return agent_memories

    def _extract_recommendation(self, investment_plan: str, final_decision: str) -> str:
        """
        Extract agent's recommendation from investment plan and final decision.

        Returns: "buy", "sell", "hold", or "unclear"
        """
        combined_text = (investment_plan + " " + final_decision).lower()

        # Check for clear buy/sell/hold signals
        if re.search(r'\b(strong buy|buy|long position|bullish|recommend buying)\b', combined_text):
            return "buy"
        elif re.search(r'\b(strong sell|sell|short position|bearish|recommend selling)\b', combined_text):
            return "sell"
        elif re.search(r'\b(hold|neutral|wait|avoid)\b', combined_text):
            return "hold"
        else:
            return "unclear"

    def _compute_correctness(self, recommendation: str, actual_direction: str) -> bool:
        """
        Determine if the agent's recommendation matched the actual outcome.

        Args:
            recommendation: "buy", "sell", "hold", or "unclear"
            actual_direction: "up" or "down"

        Returns:
            True if agent was correct, False otherwise
        """
        if recommendation == "buy" and actual_direction == "up":
            return True
        elif recommendation == "sell" and actual_direction == "down":
            return True
        elif recommendation == "hold":
            # Hold is considered neutral, so not correct/incorrect for big moves
            return False
        else:
            return False

    def _format_signals(self, signals: Dict[str, Any]) -> str:
        """Format structured signals for display."""
        lines = []
        for key, value in signals.items():
            lines.append(f"  - {key}: {value}")
        return "\n".join(lines)

    def _sample_high_movers(
        self,
        high_movers: List[Dict[str, Any]],
        max_samples: int,
        strategy: str
    ) -> List[Dict[str, Any]]:
        """
        Sample high movers based on strategy to reduce analysis time.

        Args:
            high_movers: List of all high movers found
            max_samples: Maximum number to return
            strategy: Sampling strategy (diverse, largest, recent, random)

        Returns:
            Sampled list of high movers
        """
        import random

        if len(high_movers) <= max_samples:
            return high_movers

        if strategy == "diverse":
            # Get balanced mix of up/down moves across different magnitudes
            up_moves = [m for m in high_movers if m['direction'] == 'up']
            down_moves = [m for m in high_movers if m['direction'] == 'down']

            # Sort each by magnitude
            up_moves.sort(key=lambda x: abs(x['move_pct']), reverse=True)
            down_moves.sort(key=lambda x: abs(x['move_pct']), reverse=True)

            # Take half from each direction (or proportional if imbalanced)
            up_count = min(len(up_moves), max_samples // 2)
            down_count = min(len(down_moves), max_samples - up_count)

            # If one side has fewer, take more from the other
            if up_count < max_samples // 2:
                down_count = min(len(down_moves), max_samples - up_count)
            if down_count < max_samples - up_count:
                up_count = min(len(up_moves), max_samples - down_count)

            # Stratified sampling - take from different magnitude ranges
            def stratified_sample(moves, count):
                if len(moves) <= count:
                    return moves

                # Divide into 3 buckets by magnitude
                bucket_size = len(moves) // 3
                large = moves[:bucket_size]
                medium = moves[bucket_size:bucket_size*2]
                small = moves[bucket_size*2:]

                # Sample proportionally from each bucket
                samples = []
                samples.extend(large[:count // 3])
                samples.extend(medium[:count // 3])
                samples.extend(small[:count - (2 * (count // 3))])
                return samples

            sampled = []
            sampled.extend(stratified_sample(up_moves, up_count))
            sampled.extend(stratified_sample(down_moves, down_count))

            return sampled

        elif strategy == "largest":
            # Take the largest absolute moves
            sorted_movers = sorted(high_movers, key=lambda x: abs(x['move_pct']), reverse=True)
            return sorted_movers[:max_samples]

        elif strategy == "recent":
            # Take the most recent moves
            sorted_movers = sorted(high_movers, key=lambda x: x['move_end_date'], reverse=True)
            return sorted_movers[:max_samples]

        elif strategy == "random":
            # Random sampling
            return random.sample(high_movers, max_samples)

        else:
            # Default to diverse
            return self._sample_high_movers(high_movers, max_samples, "diverse")

    def _get_stock_data_for_period(self, ticker: str, date: str) -> Dict[str, str]:
        """Gather all available data for a stock on a specific date.

        Args:
            ticker: Stock ticker symbol
            date: Date in YYYY-MM-DD format

        Returns:
            Dictionary with market_report, news_report, sentiment_report, fundamentals_report
        """
        data = {}

        try:
            # Get technical/price data (what Market Analyst sees)
            stock_data = execute_tool("get_stock_data", symbol=ticker, start_date=date)
            indicators = execute_tool("get_indicators", symbol=ticker, curr_date=date)
            data["market_report"] = f"Stock Data:\n{stock_data}\n\nTechnical Indicators:\n{indicators}"
        except Exception as e:
            data["market_report"] = f"Error fetching market data: {e}"

        try:
            # Get news (what News Analyst sees)
            news = execute_tool("get_news", symbol=ticker, from_date=date, to_date=date)
            data["news_report"] = news
        except Exception as e:
            data["news_report"] = f"Error fetching news: {e}"

        try:
            # Get sentiment (what Social Analyst sees)
            sentiment = execute_tool("get_reddit_discussions", symbol=ticker, from_date=date, to_date=date)
            data["sentiment_report"] = sentiment
        except Exception as e:
            data["sentiment_report"] = f"Error fetching sentiment: {e}"

        try:
            # Get fundamentals (what Fundamentals Analyst sees)
            fundamentals = execute_tool("get_fundamentals", symbol=ticker)
            data["fundamentals_report"] = fundamentals
        except Exception as e:
            data["fundamentals_report"] = f"Error fetching fundamentals: {e}"

        return data

    def _calculate_returns(self, ticker: str, start_date: str, end_date: str) -> Optional[float]:
        """Calculate stock returns between two dates.

        Args:
            ticker: Stock ticker symbol
            start_date: Starting date (YYYY-MM-DD)
            end_date: Ending date (YYYY-MM-DD)

        Returns:
            Percentage return, or None if data unavailable
        """
        try:
            # Get stock prices for both dates
            start_data = execute_tool("get_stock_data", symbol=ticker, start_date=start_date, end_date=start_date)
            end_data = execute_tool("get_stock_data", symbol=ticker, start_date=end_date, end_date=end_date)

            # Parse prices (this is simplified - you'd need to parse the actual response)
            # Assuming response has close price - adjust based on actual API response
            import re
            start_match = re.search(r'Close[:\s]+\$?([\d.]+)', str(start_data))
            end_match = re.search(r'Close[:\s]+\$?([\d.]+)', str(end_data))

            if start_match and end_match:
                start_price = float(start_match.group(1))
                end_price = float(end_match.group(1))
                return ((end_price - start_price) / start_price) * 100

            return None
        except Exception as e:
            print(f"Error calculating returns: {e}")
            return None

    def _create_bull_researcher_memory(self, situation: str, returns: float, ticker: str, date: str) -> str:
        """Create memory for bull researcher based on outcome.

        Returns lesson learned from bullish perspective.
        """
        if returns > 5:
            return f"""SUCCESSFUL BULLISH ANALYSIS for {ticker} on {date}:
The market conditions indicated strong bullish signals, and the stock delivered {returns:.2f}% returns.

Key takeaways:
- When similar conditions appear (strong fundamentals + positive sentiment + bullish technicals), aggressive BUY positions are warranted
- The combination of factors in this situation was a reliable indicator of upward momentum
- Continue to weight these signals heavily in future bullish arguments

Recommendation: In similar situations, advocate strongly for BUY positions with high conviction.
"""
        elif returns < -5:
            return f"""INCORRECT BULLISH SIGNALS for {ticker} on {date}:
Despite apparent bullish indicators, the stock declined {abs(returns):.2f}%.

Lessons learned:
- The bullish signals in this situation were misleading or outweighed by hidden risks
- Need to look deeper at: macro conditions, sector headwinds, or fundamental weaknesses that weren't apparent
- Be more cautious when similar patterns appear; consider bear arguments more seriously

Recommendation: In similar situations, temper bullish enthusiasm and scrutinize fundamentals more carefully.
"""
        else:
            return f"""NEUTRAL OUTCOME for {ticker} on {date}:
Stock moved {returns:.2f}%, indicating mixed signals.

Lesson: This pattern of indicators doesn't provide strong directional conviction. Look for clearer signals before making strong bullish arguments.
"""

    def _create_bear_researcher_memory(self, situation: str, returns: float, ticker: str, date: str) -> str:
        """Create memory for bear researcher based on outcome."""
        if returns < -5:
            return f"""SUCCESSFUL BEARISH ANALYSIS for {ticker} on {date}:
Bearish indicators correctly predicted decline of {abs(returns):.2f}%.

Key takeaways:
- The risk factors identified were valid and material
- Similar warning signs should be treated seriously in future analysis
- When these patterns appear, advocate strongly for SELL or reduce positions

Recommendation: In similar situations, maintain bearish stance with high conviction.
"""
        elif returns > 5:
            return f"""INCORRECT BEARISH SIGNALS for {ticker} on {date}:
Despite bearish indicators, stock rallied {returns:.2f}%.

Lessons learned:
- The bearish concerns were either overstated or offset by stronger positive factors
- Market sentiment or momentum can override fundamental concerns in short term
- Need to better assess whether bearish factors are already priced in

Recommendation: In similar situations, be more cautious about strong SELL recommendations.
"""
        else:
            return f"""NEUTRAL OUTCOME for {ticker} on {date}:
Stock moved {returns:.2f}%, mixed signals.

Lesson: These indicators don't provide clear bearish conviction. Need stronger warning signs for definitive bearish stance.
"""

    def _create_trader_memory(self, situation: str, returns: float, ticker: str, date: str) -> str:
        """Create memory for trader based on outcome."""
        if abs(returns) < 2:
            action = "HOLD"
            result = "correct - low volatility"
        elif returns > 5:
            action = "BUY"
            result = "would have been optimal"
        elif returns < -5:
            action = "SELL or avoid"
            result = "would have been optimal"
        else:
            action = "modest position"
            result = "moderate returns"

        return f"""TRADING OUTCOME for {ticker} on {date}:
Stock returned {returns:.2f}% over the evaluation period.

Optimal action: {action} - {result}

Market conditions at the time:
{situation[:500]}...

Trading lesson:
- When similar market conditions appear, consider {action} strategy
- Risk/reward profile: {'Favorable' if abs(returns) > 3 else 'Neutral'}
- Position sizing: {'Aggressive' if abs(returns) > 7 else 'Moderate' if abs(returns) > 3 else 'Conservative'}

Recommendation: Pattern recognition suggests {action} in similar future scenarios.
"""

    def _create_invest_judge_memory(self, situation: str, returns: float, ticker: str, date: str) -> str:
        """Create memory for investment judge/research manager."""
        if returns > 5:
            verdict = "Strong BUY recommendation was warranted"
        elif returns > 2:
            verdict = "Moderate BUY recommendation was appropriate"
        elif returns < -5:
            verdict = "SELL or AVOID recommendation was warranted"
        elif returns < -2:
            verdict = "HOLD or reduce exposure was appropriate"
        else:
            verdict = "HOLD recommendation was appropriate"

        return f"""INVESTMENT DECISION REVIEW for {ticker} on {date}:
Actual outcome: {returns:.2f}% return

Optimal decision: {verdict}

When synthesizing bull/bear arguments in similar conditions:
- Weight the arguments based on which perspective proved more accurate
- {"Bull arguments were stronger" if returns > 0 else "Bear arguments were stronger"}
- Factor reliability: {'High' if abs(returns) > 5 else 'Medium' if abs(returns) > 2 else 'Low'}

Recommendation for similar situations: {verdict}
"""

    def _create_risk_manager_memory(self, situation: str, returns: float, ticker: str, date: str) -> str:
        """Create memory for risk manager."""
        volatility = "HIGH" if abs(returns) > 10 else "MEDIUM" if abs(returns) > 5 else "LOW"

        if abs(returns) > 10:
            risk_assessment = "High risk - extreme volatility observed"
        elif abs(returns) > 5:
            risk_assessment = "Moderate risk - significant movement"
        else:
            risk_assessment = "Low risk - stable price action"

        return f"""RISK ASSESSMENT REVIEW for {ticker} on {date}:
Observed volatility: {volatility} (actual return: {returns:.2f}%)

Risk factors that materialized:
- Price volatility: {volatility}
- Directional risk: {'Significant downside' if returns < -5 else 'Significant upside' if returns > 5 else 'Minimal'}

Risk management lesson:
In similar market conditions:
- Position size: {'Small (high risk)' if abs(returns) > 10 else 'Moderate' if abs(returns) > 5 else 'Standard'}
- Stop loss: {'Tight (±5%)' if abs(returns) > 10 else 'Standard (±7%)' if abs(returns) > 5 else 'Relaxed (±10%)'}
- Diversification: {'Critical' if abs(returns) > 10 else 'Recommended' if abs(returns) > 5 else 'Standard'}

Recommendation: {risk_assessment}
"""

    def build_memories_for_stock(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        lookforward_days: int = 7,
        interval_days: int = 30
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Build historical memories for a stock across a date range.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            lookforward_days: How many days forward to measure returns (default: 7)
            interval_days: Days between memory samples (default: 30)

        Returns:
            Dictionary mapping agent type to list of (situation, lesson) tuples
        """
        memories = {
            "bull": [],
            "bear": [],
            "trader": [],
            "invest_judge": [],
            "risk_manager": []
        }

        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        print(f"\n🧠 Building historical memories for {ticker}")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Lookforward: {lookforward_days} days")
        print(f"   Sampling interval: {interval_days} days\n")

        sample_count = 0
        while current_date <= end_dt:
            date_str = current_date.strftime("%Y-%m-%d")
            future_date_str = (current_date + timedelta(days=lookforward_days)).strftime("%Y-%m-%d")

            print(f"   📊 Sampling {date_str}...", end=" ")

            # Get historical data for this period
            data = self._get_stock_data_for_period(ticker, date_str)
            situation = f"{data['market_report']}\n\n{data['sentiment_report']}\n\n{data['news_report']}\n\n{data['fundamentals_report']}"

            # Calculate actual returns
            returns = self._calculate_returns(ticker, date_str, future_date_str)

            if returns is not None:
                print(f"Return: {returns:+.2f}%")

                # Create agent-specific memories
                memories["bull"].append((
                    situation,
                    self._create_bull_researcher_memory(situation, returns, ticker, date_str)
                ))

                memories["bear"].append((
                    situation,
                    self._create_bear_researcher_memory(situation, returns, ticker, date_str)
                ))

                memories["trader"].append((
                    situation,
                    self._create_trader_memory(situation, returns, ticker, date_str)
                ))

                memories["invest_judge"].append((
                    situation,
                    self._create_invest_judge_memory(situation, returns, ticker, date_str)
                ))

                memories["risk_manager"].append((
                    situation,
                    self._create_risk_manager_memory(situation, returns, ticker, date_str)
                ))

                sample_count += 1
            else:
                print("⚠️  No data")

            # Move to next interval
            current_date += timedelta(days=interval_days)

        print(f"\n✅ Created {sample_count} memory samples for {ticker}")
        for agent_type in memories:
            self.memories_created[agent_type] += len(memories[agent_type])

        return memories

    def populate_agent_memories(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        lookforward_days: int = 7,
        interval_days: int = 30
    ) -> Dict[str, FinancialSituationMemory]:
        """Build and populate memories for all agent types across multiple stocks.

        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for historical analysis
            end_date: End date for historical analysis
            lookforward_days: Days forward to measure returns
            interval_days: Days between samples

        Returns:
            Dictionary of populated memory instances for each agent type
        """
        # Initialize memory stores
        agent_memories = {
            "bull": FinancialSituationMemory("bull_memory", self.config),
            "bear": FinancialSituationMemory("bear_memory", self.config),
            "trader": FinancialSituationMemory("trader_memory", self.config),
            "invest_judge": FinancialSituationMemory("invest_judge_memory", self.config),
            "risk_manager": FinancialSituationMemory("risk_manager_memory", self.config)
        }

        print("=" * 70)
        print("🏗️  HISTORICAL MEMORY BUILDER")
        print("=" * 70)

        # Build memories for each ticker
        for ticker in tickers:
            memories = self.build_memories_for_stock(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                lookforward_days=lookforward_days,
                interval_days=interval_days
            )

            # Add memories to each agent's memory store
            for agent_type, memory_list in memories.items():
                if memory_list:
                    agent_memories[agent_type].add_situations(memory_list)

        # Print summary
        print("\n" + "=" * 70)
        print("📊 MEMORY CREATION SUMMARY")
        print("=" * 70)
        for agent_type, count in self.memories_created.items():
            print(f"   {agent_type.ljust(15)}: {count} memories")
        print("=" * 70 + "\n")

        return agent_memories


# Example usage
if __name__ == "__main__":
    from tradingagents.default_config import DEFAULT_CONFIG

    # Initialize builder
    builder = HistoricalMemoryBuilder(DEFAULT_CONFIG)

    # Build memories for specific stocks over past year
    tickers = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]

    memories = builder.populate_agent_memories(
        tickers=tickers,
        start_date="2024-01-01",
        end_date="2024-12-01",
        lookforward_days=7,     # 1-week returns
        interval_days=30        # Sample monthly
    )

    # Test retrieval
    test_situation = "Strong earnings beat with positive sentiment and bullish technical indicators in tech sector"

    print("\n🔍 Testing memory retrieval...")
    print(f"Query: {test_situation}\n")

    for agent_type, memory in memories.items():
        print(f"\n{agent_type.upper()} MEMORIES:")
        results = memory.get_memories(test_situation, n_matches=2)
        for i, result in enumerate(results, 1):
            print(f"\n  Match {i} (similarity: {result['similarity_score']:.2f}):")
            print(f"  {result['recommendation'][:200]}...")
