from typing import Dict, Any, List
import re
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from tradingagents.agents.utils.agent_states import DiscoveryState
from tradingagents.agents.utils.agent_utils import (
    get_news,
    get_insider_transactions,
    get_fundamentals,
    get_indicators
)
from tradingagents.tools.executor import execute_tool
from tradingagents.schemas import TickerList, TickerContextList, MarketMovers, ThemeList

class DiscoveryGraph:
    def __init__(self, config=None):
        """
        Initialize Discovery Graph.
        
        Args:
            config: Configuration dictionary
        """
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os
        
        self.config = config or {}
        
        # Initialize LLMs using the same pattern as TradingAgentsGraph
        if self.config["llm_provider"] == "openai" or self.config["llm_provider"] == "ollama" or self.config["llm_provider"] == "openrouter":
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"] == "anthropic":
            self.deep_thinking_llm = ChatAnthropic(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatAnthropic(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"] == "google":
            # Explicitly pass Google API key from environment
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set. Please add it to your .env file.")
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"], google_api_key=google_api_key)
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"], google_api_key=google_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")
        
        # Extract discovery settings with defaults
        discovery_config = self.config.get("discovery", {})
        self.reddit_trending_limit = discovery_config.get("reddit_trending_limit", 15)
        self.market_movers_limit = discovery_config.get("market_movers_limit", 10)
        self.max_candidates_to_analyze = discovery_config.get("max_candidates_to_analyze", 10)
        self.news_lookback_days = discovery_config.get("news_lookback_days", 7)
        self.final_recommendations = discovery_config.get("final_recommendations", 3)
        
        # Store run directory for saving results
        self.run_dir = self.config.get("discovery_run_dir", None)
        
        self.graph = self._create_graph()

    def _log_tool_call(self, tool_logs: list, node: str, step_name: str, tool_name: str, params: dict, output: str, context: str = ""):
        """Log a tool call with metadata for debugging and analysis."""
        from datetime import datetime
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "node": node,
            "step": step_name,
            "tool": tool_name,
            "parameters": params,
            "context": context,
            "output": output[:1000] + "..." if len(output) > 1000 else output,
            "output_length": len(output)
        }
        tool_logs.append(log_entry)
        return log_entry

    def _save_results(self, state: dict, trade_date: str):
        """Save discovery results and tool logs to files."""
        from pathlib import Path
        from datetime import datetime
        import json
        
        # Get or create results directory
        if self.run_dir:
            results_dir = Path(self.run_dir)
        else:
            run_timestamp = datetime.now().strftime("%H_%M_%S")
            results_dir = Path(self.config.get("results_dir", "./results")) / "discovery" / trade_date / f"run_{run_timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results as markdown
        try:
            with open(results_dir / "discovery_results.md", "w") as f:
                f.write(f"# Discovery Results - {trade_date}\n\n")
                f.write(f"## Final Ranking\n\n")
                f.write(state.get("final_ranking", "No ranking available"))
                f.write("\n\n## Candidates Analyzed\n\n")
                for opp in state.get("opportunities", []):
                    f.write(f"### {opp['ticker']} ({opp['strategy']})\n\n")
        except Exception as e:
            print(f"   Error saving results: {e}")
        
        # Save as JSON
        try:
            with open(results_dir / "discovery_result.json", "w") as f:
                json_state = {
                    "trade_date": trade_date,
                    "tickers": state.get("tickers", []),
                    "filtered_tickers": state.get("filtered_tickers", []),
                    "final_ranking": state.get("final_ranking", ""),
                    "status": state.get("status", "")
                }
                json.dump(json_state, f, indent=2)
        except Exception as e:
            print(f"   Error saving JSON: {e}")
        
        # Save tool logs
        tool_logs = state.get("tool_logs", [])
        if tool_logs:
            try:
                with open(results_dir / "tool_execution_logs.json", "w") as f:
                    json.dump(tool_logs, f, indent=2)
                
                with open(results_dir / "tool_execution_logs.md", "w") as f:
                    f.write(f"# Tool Execution Logs - {trade_date}\n\n")
                    for i, log in enumerate(tool_logs, 1):
                        f.write(f"## {i}. {log['step']}\n\n")
                        f.write(f"- **Tool:** `{log['tool']}`\n")
                        f.write(f"- **Node:** {log['node']}\n")
                        f.write(f"- **Timestamp:** {log['timestamp']}\n")
                        if log.get('context'):
                            f.write(f"- **Context:** {log['context']}\n")
                        f.write(f"- **Parameters:** `{log['parameters']}`\n")
                        f.write(f"- **Output Length:** {log['output_length']} chars\n\n")
                        f.write(f"### Output\n```\n{log['output']}\n```\n\n")
                        f.write("---\n\n")
            except Exception as e:
                print(f"   Error saving tool logs: {e}")
        
        print(f"   Results saved to: {results_dir}")

    def _create_graph(self):
        workflow = StateGraph(DiscoveryState)

        workflow.add_node("scanner", self.scanner_node)
        workflow.add_node("filter", self.filter_node)
        workflow.add_node("deep_dive", self.deep_dive_node)
        workflow.add_node("ranker", self.ranker_node)

        workflow.set_entry_point("scanner")
        workflow.add_edge("scanner", "filter")
        workflow.add_edge("filter", "deep_dive")
        workflow.add_edge("deep_dive", "ranker")
        workflow.add_edge("ranker", END)

        return workflow.compile()

    def scanner_node(self, state: DiscoveryState):
        """Scan the market for potential candidates."""
        print("🔍 Scanning market for opportunities...")
        
        candidates = []
        tool_logs = state.get("tool_logs", [])
        
        # 0. Macro Theme Discovery (Top-Down) - DISABLED
        # This section used Twitter API which has rate limit issues
        # try:
        #     from datetime import datetime
        #     today = datetime.now().strftime("%Y-%m-%d")
        #     global_news = execute_tool("get_global_news", date=today, limit=5)
        #     ... (macro theme code disabled)
        # except Exception as e:
        #     print(f"   Error in Macro Theme Discovery: {e}")

        # 1. Get Reddit Trending (Social Sentiment)
        try:
            reddit_report = execute_tool("get_trending_tickers", limit=self.reddit_trending_limit)
            # Use LLM to extract tickers WITH context
            prompt = """Extract valid stock ticker symbols from this Reddit report, along with context about why they're trending.

For each ticker, include:
- ticker: The stock symbol (1-5 uppercase letters)
- context: Brief description of sentiment, mentions, or key discussion points

Do not include currencies (RMB), cryptocurrencies (BTC), or invalid symbols.

Report:
{report}

Return a JSON object with a 'candidates' array of objects, each having 'ticker' and 'context' fields.""".format(report=reddit_report)
            
            # Use structured output for ticker+context extraction
            structured_llm = self.quick_thinking_llm.with_structured_output(
                schema=TickerContextList.model_json_schema(),
                method="json_schema"
            )
            response = structured_llm.invoke([HumanMessage(content=prompt)])
            
            # Validate and add tickers with context
            reddit_candidates = response.get("candidates", [])
            for c in reddit_candidates:
                ticker = c.get("ticker", "").upper().strip()
                context = c.get("context", "Trending on Reddit")
                # Validate ticker - Exclude garbage, verify existence
                if re.match(r'^[A-Z]{1,5}$', ticker):
                    try:
                        if execute_tool("validate_ticker", symbol=ticker):
                             candidates.append({"ticker": ticker, "source": "social_trending", "context": context})
                    except: pass
        except Exception as e:
            print(f"   Error fetching Reddit tickers: {e}")

        # 2. Get Twitter Trending (Social Sentiment) - DISABLED due to API issues
        # try:
        #     # Search for general market discussions
        #     tweets_report = execute_tool("get_tweets", query="stocks to watch", count=20)
        #     
        #     # Use LLM to extract tickers
        #     prompt = """Extract ONLY valid stock ticker symbols from this Twitter report.
        # ... (Twitter extraction code disabled)
        # except Exception as e:
        #     print(f"   Error fetching Twitter tickers: {e}")

        # 2. Get Market Movers (Gainers & Losers)
        try:
            movers_report = execute_tool("get_market_movers", limit=self.market_movers_limit)
            # Use LLM to extract movers with context
            prompt = f"""Extract stock tickers from this market movers data with context about their performance.

For each ticker, include:
- ticker: The stock symbol (1-5 uppercase letters)
- type: Either 'gainer' or 'loser'
- reason: Brief description of the price movement (%, volume, catalyst if mentioned)

Data:
{movers_report}

Return a JSON object with a 'movers' array containing objects with 'ticker', 'type', and 'reason' fields."""
            
            # Use structured output for market movers
            structured_llm = self.quick_thinking_llm.with_structured_output(
                schema=MarketMovers.model_json_schema(),
                method="json_schema"
            )
            response = structured_llm.invoke([HumanMessage(content=prompt)])
            
            # Validate and add tickers with context
            movers = response.get("movers", [])
            for m in movers:
                ticker = m.get('ticker', '').upper().strip()
                if ticker and re.match(r'^[A-Z]{1,5}$', ticker):
                    try:
                        if execute_tool("validate_ticker", symbol=ticker):
                            mover_type = m.get('type', 'gainer')
                            reason = m.get('reason', f"Top {mover_type}")
                            candidates.append({
                                "ticker": ticker, 
                                "source": "market_mover", 
                                "context": f"{reason} ({m.get('change_percent', 0)}%)"
                            })
                    except: pass

        except Exception as e:
            print(f"   Error fetching Market Movers: {e}")

        # 3. Get Earnings Calendar (Event-based Discovery)
        try:
            from datetime import datetime, timedelta
            today = datetime.now()
            from_date = today.strftime("%Y-%m-%d")
            to_date = (today + timedelta(days=7)).strftime("%Y-%m-%d")  # Next 7 days

            earnings_report = execute_tool("get_earnings_calendar", from_date=from_date, to_date=to_date)

            # Extract tickers with earnings context
            prompt = """Extract stock tickers from this earnings calendar with context about their upcoming earnings.

For each ticker, include:
- ticker: The stock symbol (1-5 uppercase letters)
- context: Earnings date, expected EPS, and any other relevant info

Earnings Calendar:
{report}

Return a JSON object with a 'candidates' array of objects, each having 'ticker' and 'context' fields.""".format(report=earnings_report)

            structured_llm = self.quick_thinking_llm.with_structured_output(
                schema=TickerContextList.model_json_schema(),
                method="json_schema"
            )
            response = structured_llm.invoke([HumanMessage(content=prompt)])

            earnings_candidates = response.get("candidates", [])
            for c in earnings_candidates:
                ticker = c.get("ticker", "").upper().strip()
                context = c.get("context", "Upcoming earnings")
                if re.match(r'^[A-Z]{1,5}$', ticker):
                     try:
                        if execute_tool("validate_ticker", symbol=ticker):
                            candidates.append({"ticker": ticker, "source": "earnings_catalyst", "context": context})
                     except: pass
        except Exception as e:
            print(f"   Error fetching Earnings Calendar: {e}")

        # 4. Get IPO Calendar (New Listings Discovery)
        try:
            from datetime import datetime, timedelta
            today = datetime.now()
            from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")  # Past 7 days
            to_date = (today + timedelta(days=14)).strftime("%Y-%m-%d")   # Next 14 days

            ipo_report = execute_tool("get_ipo_calendar", from_date=from_date, to_date=to_date)

            # Extract tickers with IPO context
            prompt = """Extract stock tickers from this IPO calendar with context about the offering.

For each ticker, include:
- ticker: The stock symbol (1-5 uppercase letters)
- context: IPO date, price range, shares offered, and company description

IPO Calendar:
{report}

Return a JSON object with a 'candidates' array of objects, each having 'ticker' and 'context' fields.""".format(report=ipo_report)

            structured_llm = self.quick_thinking_llm.with_structured_output(
                schema=TickerContextList.model_json_schema(),
                method="json_schema"
            )
            response = structured_llm.invoke([HumanMessage(content=prompt)])

            ipo_candidates = response.get("candidates", [])
            for c in ipo_candidates:
                ticker = c.get("ticker", "").upper().strip()
                context = c.get("context", "Recent/upcoming IPO")
                if re.match(r'^[A-Z]{1,5}$', ticker):
                     try:
                        if execute_tool("validate_ticker", symbol=ticker):
                            candidates.append({"ticker": ticker, "source": "ipo_listing", "context": context})
                     except: pass
        except Exception as e:
            print(f"   Error fetching IPO Calendar: {e}")

        # 5. Short Squeeze Detection (High Short Interest)
        try:
            # Get stocks with high short interest - potential squeeze candidates
            short_interest_report = execute_tool(
                "get_short_interest",
                min_short_interest_pct=15.0,  # 15%+ short interest
                min_days_to_cover=3.0,        # 3+ days to cover
                top_n=15
            )
            
            # Extract tickers with short squeeze context
            prompt = """Extract stock tickers from this short interest report with context about squeeze potential.

For each ticker, include:
- ticker: The stock symbol (1-5 uppercase letters)
- context: Short interest %, days to cover, squeeze potential rating, and any other relevant metrics

Short Interest Report:
{report}

Return a JSON object with a 'candidates' array of objects, each having 'ticker' and 'context' fields.""".format(report=short_interest_report)

            structured_llm = self.quick_thinking_llm.with_structured_output(
                schema=TickerContextList.model_json_schema(),
                method="json_schema"
            )
            response = structured_llm.invoke([HumanMessage(content=prompt)])

            short_candidates = response.get("candidates", [])
            for c in short_candidates:
                ticker = c.get("ticker", "").upper().strip()
                context = c.get("context", "High short interest")
                if re.match(r'^[A-Z]{1,5}$', ticker):
                     try:
                        if execute_tool("validate_ticker", symbol=ticker):
                            candidates.append({"ticker": ticker, "source": "short_squeeze", "context": context})
                     except: pass
            
            print(f"   Found {len(short_candidates)} short squeeze candidates")
        except Exception as e:
            print(f"   Error fetching Short Interest: {e}")

        # 6. Unusual Volume Detection (Accumulation Signal)
        try:
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            
            volume_report = execute_tool(
                "get_unusual_volume",
                date=today,
                min_volume_multiple=3.0,  # 3x average volume
                max_price_change=5.0,     # Less than 5% price change
                top_n=15
            )
            
            # Extract tickers with volume context
            prompt = """Extract stock tickers from this unusual volume report with context about the accumulation pattern.

For each ticker, include:
- ticker: The stock symbol (1-5 uppercase letters)
- context: Volume multiple, price change, and any interpretation of the pattern

Unusual Volume Report:
{report}

Return a JSON object with a 'candidates' array of objects, each having 'ticker' and 'context' fields.""".format(report=volume_report)

            structured_llm = self.quick_thinking_llm.with_structured_output(
                schema=TickerContextList.model_json_schema(),
                method="json_schema"
            )
            response = structured_llm.invoke([HumanMessage(content=prompt)])

            volume_candidates = response.get("candidates", [])
            for c in volume_candidates:
                ticker = c.get("ticker", "").upper().strip()
                context = c.get("context", "Unusual volume pattern")
                if re.match(r'^[A-Z]{1,5}$', ticker):
                     try:
                        if execute_tool("validate_ticker", symbol=ticker):
                            candidates.append({"ticker": ticker, "source": "unusual_volume", "context": context})
                     except: pass
            
            print(f"   Found {len(volume_candidates)} unusual volume candidates")
        except Exception as e:
            print(f"   Error fetching Unusual Volume: {e}")

        # 7. Analyst Rating Changes (Institutional Catalyst)
        try:
            analyst_report = execute_tool(
                "get_analyst_rating_changes",
                lookback_days=7,
                change_types=["upgrade", "initiated"],  # Focus on positive catalysts
                top_n=15
            )
            
            # Extract tickers with analyst context
            prompt = """Extract stock tickers from this analyst rating changes report with context about the rating action.

For each ticker, include:
- ticker: The stock symbol (1-5 uppercase letters)
- context: Type of change (upgrade/initiated), analyst firm, price target, and any other relevant details

Analyst Rating Changes:
{report}

Return a JSON object with a 'candidates' array of objects, each having 'ticker' and 'context' fields.""".format(report=analyst_report)

            structured_llm = self.quick_thinking_llm.with_structured_output(
                schema=TickerContextList.model_json_schema(),
                method="json_schema"
            )
            response = structured_llm.invoke([HumanMessage(content=prompt)])

            analyst_candidates = response.get("candidates", [])
            for c in analyst_candidates:
                ticker = c.get("ticker", "").upper().strip()
                context = c.get("context", "Recent analyst action")
                if re.match(r'^[A-Z]{1,5}$', ticker):
                    try:
                        if execute_tool("validate_ticker", symbol=ticker):
                            candidates.append({"ticker": ticker, "source": "analyst_upgrade", "context": context})
                    except: pass
            
            print(f"   Found {len(analyst_candidates)} analyst upgrade candidates")
        except Exception as e:
            print(f"   Error fetching Analyst Ratings: {e}")

        # Deduplicate
        unique_candidates = {}
        for c in candidates:
            if c['ticker'] not in unique_candidates:
                unique_candidates[c['ticker']] = c
        
        final_candidates = list(unique_candidates.values())
        print(f"   Found {len(final_candidates)} unique candidates.")
        return {"tickers": [c['ticker'] for c in final_candidates], "candidate_metadata": final_candidates, "tool_logs": tool_logs, "status": "scanned"}

    def filter_node(self, state: DiscoveryState):
        """Filter candidates based on strategy (Contrarian vs Momentum)."""
        candidates = state.get("candidate_metadata", [])
        if not candidates:
            # Fallback if metadata missing (backward compatibility)
            candidates = [{"ticker": t, "source": "unknown"} for t in state["tickers"]]
            
        print(f"🔍 Filtering {len(candidates)} candidates...")
        
        filtered_candidates = []
        
        for cand in candidates:
            ticker = cand['ticker']
            source = cand['source']
            
            try:
                # Get Fundamentals
                # We use get_fundamentals to get P/E, Market Cap, etc.
                # Since get_fundamentals returns a JSON string (from Alpha Vantage), we can parse it.
                # Note: In a real run, we'd use the tool. Here we simulate the logic.
                
                # Logic:
                # 1. Contrarian (Losers): Look for Strong Fundamentals (Low P/E, High Profit)
                # 2. Momentum (Gainers/Social): Look for Growth (Revenue Growth)
                
                # For this implementation, we'll pass them to the deep dive 
                # but tag them with the strategy we want to verify.
                
                strategy = "momentum"
                if source == "loser":
                    strategy = "contrarian_value"
                elif source == "social_trending" or source == "twitter_sentiment":
                    strategy = "social_hype"
                elif source == "earnings_catalyst":
                    strategy = "earnings_play"
                elif source == "ipo_listing":
                    strategy = "ipo_opportunity"
                
                cand['strategy'] = strategy
                
                # Technical Analysis Check (New)
                try:
                    from datetime import datetime
                    today = datetime.now().strftime("%Y-%m-%d")
                    
                    # Get RSI (and other indicators)
                    rsi_data = execute_tool("get_indicators", symbol=ticker, curr_date=today)
                    
                    # Simple parsing of the string report to find the latest value
                    # The report format is usually "## rsi values...\n\nDATE: VALUE"
                    # We'll just store the report for the LLM to analyze in deep dive if needed, 
                    # OR we can try to parse it here. For now, let's just add it to metadata.
                    cand['technical_indicators'] = rsi_data
                    
                except Exception as e:
                    print(f"   Error getting technicals for {ticker}: {e}")
                
                filtered_candidates.append(cand)
                
            except Exception as e:
                print(f"   Error checking {ticker}: {e}")
        
        # Limit to configured max
        filtered_candidates = filtered_candidates[:self.max_candidates_to_analyze]
        
        print(f"   Selected {len(filtered_candidates)} for deep dive.")
        return {"filtered_tickers": [c['ticker'] for c in filtered_candidates], "candidate_metadata": filtered_candidates, "status": "filtered"}

    def deep_dive_node(self, state: DiscoveryState):
        """Perform deep dive analysis on selected candidates."""
        candidates = state.get("candidate_metadata", [])
        trade_date = state.get("trade_date", "")
        
        # Calculate date range for news (configurable days back from trade_date)
        from datetime import datetime, timedelta
        
        if trade_date:
            end_date_obj = datetime.strptime(trade_date, "%Y-%m-%d")
        else:
            end_date_obj = datetime.now()
            
        start_date_obj = end_date_obj - timedelta(days=self.news_lookback_days)
        start_date = start_date_obj.strftime("%Y-%m-%d")
        end_date = end_date_obj.strftime("%Y-%m-%d")
        
        print(f"🔍 Performing deep dive on {len(candidates)} candidates...")
        print(f"   News date range: {start_date} to {end_date}")
        
        opportunities = []
        
        for cand in candidates:
            ticker = cand['ticker']
            strategy = cand['strategy']
            print(f"   Analyzing {ticker} ({strategy})...")
            
            try:
                # 1. Get News Sentiment
                news = execute_tool("get_news", ticker=ticker, start_date=start_date, end_date=end_date)
                
                # 2. Get Insider Transactions & Sentiment
                insider = execute_tool("get_insider_transactions", ticker=ticker)
                insider_sentiment = execute_tool("get_insider_sentiment", ticker=ticker)
                
                # 3. Get Fundamentals (for the Contrarian check)
                fundamentals = execute_tool("get_fundamentals", ticker=ticker, curr_date=end_date)
                
                # 4. Get Analyst Recommendations
                recommendations = execute_tool("get_recommendation_trends", ticker=ticker)
                
                opportunities.append({
                    "ticker": ticker,
                    "strategy": strategy,
                    "news": news,
                    "insider_transactions": insider,
                    "insider_sentiment": insider_sentiment,
                    "fundamentals": fundamentals,
                    "recommendations": recommendations
                })
                
            except Exception as e:
                print(f"   Failed to analyze {ticker}: {e}")
        
        return {"opportunities": opportunities, "status": "analyzed"}

    def ranker_node(self, state: DiscoveryState):
        """Rank opportunities and select the best ones."""
        from datetime import datetime
        
        opportunities = state["opportunities"]
        print("🔍 Ranking opportunities...")
        
        # Truncate data to prevent token limit errors
        # Keep only essential info for ranking
        truncated_opps = []
        for opp in opportunities:
            truncated_opps.append({
                "ticker": opp["ticker"],
                "strategy": opp["strategy"],
                # Truncate to ~1000 chars each (roughly 250 tokens)
                "news": opp["news"][:1000] + "..." if len(opp["news"]) > 1000 else opp["news"],
                "insider_sentiment": opp.get("insider_sentiment", "")[:500],
                "insider_transactions": opp["insider_transactions"][:1000] + "..." if len(opp["insider_transactions"]) > 1000 else opp["insider_transactions"],
                "fundamentals": opp["fundamentals"][:1000] + "..." if len(opp["fundamentals"]) > 1000 else opp["fundamentals"],
                "recommendations": opp["recommendations"][:1000] + "..." if len(opp["recommendations"]) > 1000 else opp["recommendations"],
            })
        
        prompt = f"""
        Analyze these investment opportunities and select the TOP {self.final_recommendations} most promising ones.
        
        STRATEGIES TO LOOK FOR:
        1. **Contrarian Value**: Stock is a "Loser" or has bad sentiment, BUT has strong fundamentals (Low P/E, good financials).
        2. **Momentum/Hype**: Stock is Trending/Gainer AND has news/growth to support it.
        3. **Insider Play**: Significant insider buying regardless of trend.
        
        OPPORTUNITIES:
        {truncated_opps}
        
        Return a JSON list of the top {self.final_recommendations}, with fields: 
        - "ticker"
        - "strategy_match" (e.g., "Contrarian Value", "Momentum")
        - "reason" (Explain WHY it fits the strategy)
        - "confidence" (0-10)
        """
        
        response = self.deep_thinking_llm.invoke([HumanMessage(content=prompt)])
        
        print("   Ranking complete.")
        
        # Build result state
        result_state = {
            "status": "complete", 
            "opportunities": opportunities, 
            "final_ranking": response.content,
            "tool_logs": state.get("tool_logs", [])
        }
        
        # Save results to files
        trade_date = state.get("trade_date", datetime.now().strftime("%Y-%m-%d"))
        self._save_results(result_state, trade_date)
        
        return result_state
