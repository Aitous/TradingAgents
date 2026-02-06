import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class DiscoveryAnalytics:
    """
    Handles performance tracking, statistics, and result saving for the Discovery Graph.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.recommendations_dir = self.data_dir / "recommendations"
        self.recommendations_dir.mkdir(parents=True, exist_ok=True)

    def update_performance_tracking(self):
        """Update performance metrics for all open recommendations."""
        print("📊 Updating recommendation performance tracking...")

        if not self.recommendations_dir.exists():
            print("   No historical recommendations to track yet.")
            return

        # Load all recommendations
        all_recs = []
        # Use glob directly on the path object if python 3.10+ otherwise str()
        pattern = str(self.recommendations_dir / "*.json")

        for filepath in glob.glob(pattern):
            # Skip the database and stats files
            if "performance_database" in filepath or "statistics" in filepath:
                continue

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    recs = data.get("recommendations", [])
                    for rec in recs:
                        rec["discovery_date"] = data.get(
                            "date", os.path.basename(filepath).replace(".json", "")
                        )
                        all_recs.append(rec)
            except Exception as e:
                print(f"   Warning: Error loading {filepath}: {e}")

        if not all_recs:
            print("   No recommendations found to track.")
            return

        # Filter to only track open positions
        open_recs = [r for r in all_recs if r.get("status") != "closed"]
        print(f"   Tracking {len(open_recs)} open positions (out of {len(all_recs)} total)...")

        # Update performance
        today = datetime.now().strftime("%Y-%m-%d")
        updated_count = 0

        for rec in all_recs:
            ticker = rec.get("ticker")
            discovery_date = rec.get("discovery_date")
            entry_price = rec.get("entry_price")

            # Skip if already closed or missing data
            if rec.get("status") == "closed" or not all([ticker, discovery_date, entry_price]):
                continue

            try:
                # Get current price
                # We interpret this import here to avoid circular dependency if this class is imported early
                from tradingagents.dataflows.y_finance import get_stock_price

                current_price = get_stock_price(ticker, curr_date=today)

                if current_price is None:
                    continue

                # Calculate metrics
                rec_date = datetime.strptime(discovery_date, "%Y-%m-%d")
                days_held = (datetime.now() - rec_date).days
                return_pct = ((current_price - entry_price) / entry_price) * 100

                # Update
                rec["current_price"] = current_price
                rec["return_pct"] = round(return_pct, 2)
                rec["days_held"] = days_held
                rec["last_updated"] = today

                # Capture specific time periods (1d, 7d, 30d)
                if days_held >= 1 and "return_1d" not in rec:
                    rec["return_1d"] = round(return_pct, 2)
                    rec["win_1d"] = return_pct > 0

                if days_held >= 7 and "return_7d" not in rec:
                    rec["return_7d"] = round(return_pct, 2)
                    rec["win_7d"] = return_pct > 0

                if days_held >= 30 and "return_30d" not in rec:
                    rec["return_30d"] = round(return_pct, 2)
                    rec["win_30d"] = return_pct > 0
                    rec["status"] = "closed"

                updated_count += 1

            except Exception:
                # Silently skip errors to not interrupt discovery
                pass

        if updated_count > 0:
            print(f"   Updated {updated_count} positions")
            self._save_performance_db(all_recs)
        else:
            print("   No updates needed")

    def _save_performance_db(self, all_recs: List[Dict]):
        """Save the aggregated performance database and recalculate stats."""
        # Save updated database
        by_date = {}
        for rec in all_recs:
            date = rec.get("discovery_date", "unknown")
            if date not in by_date:
                by_date[date] = []
            by_date[date].append(rec)

        db_path = self.recommendations_dir / "performance_database.json"
        with open(db_path, "w") as f:
            json.dump(
                {
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_recommendations": len(all_recs),
                    "recommendations_by_date": by_date,
                },
                f,
                indent=2,
            )

        # Calculate and save statistics
        stats = self.calculate_statistics(all_recs)
        stats_path = self.recommendations_dir / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print("   💾 Updated performance database and statistics")

    def calculate_statistics(self, recommendations: list) -> dict:
        """Calculate aggregate statistics from historical performance."""
        stats = {
            "total_recommendations": len(recommendations),
            "by_strategy": {},
            "overall_1d": {"count": 0, "wins": 0, "avg_return": 0},
            "overall_7d": {"count": 0, "wins": 0, "avg_return": 0},
            "overall_30d": {"count": 0, "wins": 0, "avg_return": 0},
        }

        # Calculate by strategy
        for rec in recommendations:
            strategy = rec.get("strategy_match", "unknown")

            if strategy not in stats["by_strategy"]:
                stats["by_strategy"][strategy] = {
                    "count": 0,
                    "wins_1d": 0,
                    "losses_1d": 0,
                    "wins_7d": 0,
                    "losses_7d": 0,
                    "wins_30d": 0,
                    "losses_30d": 0,
                    "avg_return_1d": 0,
                    "avg_return_7d": 0,
                    "avg_return_30d": 0,
                }

            stats["by_strategy"][strategy]["count"] += 1

            # 1-day stats
            if "return_1d" in rec:
                stats["overall_1d"]["count"] += 1
                if rec.get("win_1d"):
                    stats["overall_1d"]["wins"] += 1
                    stats["by_strategy"][strategy]["wins_1d"] += 1
                else:
                    stats["by_strategy"][strategy]["losses_1d"] += 1
                stats["overall_1d"]["avg_return"] += rec["return_1d"]

            # 7-day stats
            if "return_7d" in rec:
                stats["overall_7d"]["count"] += 1
                if rec.get("win_7d"):
                    stats["overall_7d"]["wins"] += 1
                    stats["by_strategy"][strategy]["wins_7d"] += 1
                else:
                    stats["by_strategy"][strategy]["losses_7d"] += 1
                stats["overall_7d"]["avg_return"] += rec["return_7d"]

            # 30-day stats
            if "return_30d" in rec:
                stats["overall_30d"]["count"] += 1
                if rec.get("win_30d"):
                    stats["overall_30d"]["wins"] += 1
                    stats["by_strategy"][strategy]["wins_30d"] += 1
                else:
                    stats["by_strategy"][strategy]["losses_30d"] += 1
                stats["overall_30d"]["avg_return"] += rec["return_30d"]

        # Calculate averages and win rates
        self._calculate_metric_averages(stats["overall_1d"])
        self._calculate_metric_averages(stats["overall_7d"])
        self._calculate_metric_averages(stats["overall_30d"])

        # Calculate per-strategy stats
        for strategy, data in stats["by_strategy"].items():
            total_1d = data["wins_1d"] + data["losses_1d"]
            total_7d = data["wins_7d"] + data["losses_7d"]
            total_30d = data["wins_30d"] + data["losses_30d"]

            if total_1d > 0:
                data["win_rate_1d"] = round((data["wins_1d"] / total_1d) * 100, 1)

            if total_7d > 0:
                data["win_rate_7d"] = round((data["wins_7d"] / total_7d) * 100, 1)

            if total_30d > 0:
                data["win_rate_30d"] = round((data["wins_30d"] / total_30d) * 100, 1)

        return stats

    def _calculate_metric_averages(self, metric_dict):
        if metric_dict["count"] > 0:
            metric_dict["win_rate"] = round((metric_dict["wins"] / metric_dict["count"]) * 100, 1)
            metric_dict["avg_return"] = round(metric_dict["avg_return"] / metric_dict["count"], 2)

    def load_historical_stats(self) -> dict:
        """Load historical performance statistics."""
        stats_file = self.recommendations_dir / "statistics.json"

        if not stats_file.exists():
            return {
                "available": False,
                "message": "No historical data yet - this will improve over time as we track performance",
            }

        try:
            with open(stats_file, "r") as f:
                stats = json.load(f)

            # Format insights
            insights = {
                "available": True,
                "total_tracked": stats.get("total_recommendations", 0),
                "overall_1d_win_rate": stats.get("overall_1d", {}).get("win_rate", 0),
                "overall_7d_win_rate": stats.get("overall_7d", {}).get("win_rate", 0),
                "overall_30d_win_rate": stats.get("overall_30d", {}).get("win_rate", 0),
                "by_strategy": stats.get("by_strategy", {}),
                "summary": self.format_stats_summary(stats),
            }

            return insights

        except Exception as e:
            print(f"   Warning: Could not load historical stats: {e}")
            return {"available": False, "message": "Error loading historical data"}

    def format_stats_summary(self, stats: dict) -> str:
        """Format statistics into a concise summary."""
        lines = []

        overall_1d = stats.get("overall_1d", {})
        overall_7d = stats.get("overall_7d", {})
        overall_30d = stats.get("overall_30d", {})

        if overall_1d.get("count", 0) > 0:
            lines.append(
                f"Historical 1-day win rate: {overall_1d.get('win_rate', 0)}% ({overall_1d.get('count')} tracked)"
            )

        if overall_7d.get("count", 0) > 0:
            lines.append(
                f"Historical 7-day win rate: {overall_7d.get('win_rate', 0)}% ({overall_7d.get('count')} tracked)"
            )

        if overall_30d.get("count", 0) > 0:
            lines.append(
                f"Historical 30-day win rate: {overall_30d.get('win_rate', 0)}% ({overall_30d.get('count')} tracked)"
            )

        # Top performing strategies
        by_strategy = stats.get("by_strategy", {})
        if by_strategy:
            lines.append("\nBest performing strategies (7-day):")
            sorted_strats = sorted(
                [(k, v) for k, v in by_strategy.items() if v.get("win_rate_7d")],
                key=lambda x: x[1].get("win_rate_7d", 0),
                reverse=True,
            )[:3]

            for strategy, data in sorted_strats:
                wr = data.get("win_rate_7d", 0)
                count = data.get("wins_7d", 0) + data.get("losses_7d", 0)
                lines.append(f"  - {strategy}: {wr}% win rate ({count} samples)")

        return "\n".join(lines) if lines else "No historical data available yet"

    def save_recommendations(self, rankings: list, trade_date: str, llm_provider: str):
        """Save recommendations for tracking."""
        from tradingagents.dataflows.y_finance import get_stock_price

        # Get current prices for entry tracking
        enriched_rankings = []
        for rank in rankings:
            ticker = rank.get("ticker")

            # Get current price as entry price
            try:
                entry_price = get_stock_price(ticker, curr_date=trade_date)
            except Exception as e:
                print(f"   Warning: Could not get entry price for {ticker}: {e}")
                entry_price = None

            enriched_rankings.append(
                {
                    "ticker": ticker,
                    "rank": rank.get("rank"),
                    "strategy_match": rank.get("strategy_match"),
                    "final_score": rank.get("final_score"),
                    "confidence": rank.get("confidence"),
                    "reason": rank.get("reason"),
                    "entry_price": entry_price,
                    "discovery_date": trade_date,
                    "status": "open",  # open or closed
                }
            )

        # Save to dated file
        output_file = self.recommendations_dir / f"{trade_date}.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "date": trade_date,
                    "llm_provider": llm_provider,
                    "recommendations": enriched_rankings,
                },
                f,
                indent=2,
            )

        print(f"   📊 Saved {len(enriched_rankings)} recommendations for tracking: {output_file}")

    def save_discovery_results(self, state: dict, trade_date: str, config: Dict[str, Any]):
        """Save full discovery results and tool logs."""

        run_dir = config.get("discovery_run_dir")
        if run_dir:
            results_dir = Path(run_dir)
        else:
            run_timestamp = datetime.now().strftime("%H_%M_%S")
            results_dir = (
                Path(config.get("results_dir", "./results"))
                / "discovery"
                / trade_date
                / f"run_{run_timestamp}"
            )
            results_dir.mkdir(parents=True, exist_ok=True)

        # Save main results as markdown
        try:
            with open(results_dir / "discovery_results.md", "w") as f:
                f.write(f"# Discovery Analysis - {trade_date}\n\n")
                f.write(f"**LLM Provider**: {config.get('llm_provider', 'unknown').upper()}\n")
                f.write(
                    f"**Models**: Shallow={config.get('quick_think_llm', 'N/A')}, Deep={config.get('deep_think_llm', 'N/A')}\n\n"
                )
                f.write("## Top Investment Opportunities\n\n")

                final_ranking = state.get("final_ranking", "")
                if final_ranking:
                    self._write_ranking_md(f, final_ranking)
                else:
                    f.write("*No recommendations generated.*\n\n")

                # Format candidates analyzed section
                f.write("\n## All Candidates Analyzed\n\n")
                opportunities = state.get("opportunities", [])
                if opportunities:
                    f.write(f"Total candidates analyzed: {len(opportunities)}\n\n")
                    for opp in opportunities:
                        ticker = opp.get("ticker", "UNKNOWN")
                        strategy = opp.get("strategy", "N/A")
                        f.write(f"- **{ticker}** ({strategy})\n")

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
                    "status": state.get("status", ""),
                }
                json.dump(json_state, f, indent=2)
        except Exception as e:
            print(f"   Error saving JSON: {e}")

        # Save tool logs
        tool_logs = state.get("tool_logs", [])
        if tool_logs:
            tool_log_max_chars = (
                config.get("discovery", {}).get("tool_log_max_chars", 10_000)
                if config
                else 10_000
            )
            self._save_tool_logs(results_dir, tool_logs, trade_date, tool_log_max_chars)

        print(f"   Results saved to: {results_dir}")

    def _write_ranking_md(self, f, final_ranking):
        try:
            # Handle both string and dict/list formats
            if isinstance(final_ranking, str):
                rankings = json.loads(final_ranking)
            else:
                rankings = final_ranking

            # Handle both direct list and dict with 'rankings' key
            if isinstance(rankings, dict):
                rankings = rankings.get("rankings", [])

            for rank in rankings:
                ticker = rank.get("ticker", "UNKNOWN")
                company_name = rank.get("company_name", ticker)
                current_price = rank.get("current_price")
                description = rank.get("description", "")
                strategy = rank.get("strategy_match", "N/A")
                final_score = rank.get("final_score", 0)
                confidence = rank.get("confidence", 0)
                reason = rank.get("reason", "")
                rank_num = rank.get("rank", "?")

                # Format price
                price_str = f"${current_price:.2f}" if current_price else "N/A"

                # Write formatted recommendation
                f.write(f"### #{rank_num}: {ticker}\n\n")
                f.write(f"**Company:** {company_name}\n\n")
                f.write(f"**Current Price:** {price_str}\n\n")
                f.write(f"**Strategy:** {strategy}\n\n")
                f.write(f"**Score:** {final_score} | **Confidence:** {confidence}/10\n\n")

                if description:
                    f.write("**Description:**\n\n")
                    f.write(f"> {description}\n\n")

                f.write("**Investment Thesis:**\n\n")
                # Wrap long text nicely
                wrapped_reason = reason.replace(". ", ".\n\n")
                f.write(f"{wrapped_reason}\n\n")
                f.write("---\n\n")
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            f.write(f"⚠️ Error formatting rankings: {e}\n\n")
            f.write("```json\n")
            f.write(str(final_ranking))
            f.write("\n```\n\n")

    def _save_tool_logs(
        self, results_dir: Path, tool_logs: list, trade_date: str, md_max_chars: int
    ):
        try:
            with open(results_dir / "tool_execution_logs.json", "w") as f:
                json.dump(tool_logs, f, indent=2)

            with open(results_dir / "tool_execution_logs.md", "w") as f:
                f.write(f"# Tool Execution Logs - {trade_date}\n\n")
                for i, log in enumerate(tool_logs, 1):
                    step = log.get("step", "Unknown step")
                    log_type = log.get("type", "tool")
                    f.write(f"## {i}. {step}\n\n")
                    f.write(f"- **Type:** `{log_type}`\n")
                    f.write(f"- **Node:** {log.get('node', '')}\n")
                    f.write(f"- **Timestamp:** {log.get('timestamp', '')}\n")
                    if log.get("context"):
                        f.write(f"- **Context:** {log['context']}\n")
                    if log.get("error"):
                        f.write(f"- **Error:** {log['error']}\n")

                    if log_type == "llm":
                        f.write(f"- **Model:** `{log.get('model', 'unknown')}`\n")
                        f.write(f"- **Prompt Length:** {log.get('prompt_length', 0)} chars\n")
                        f.write(f"- **Output Length:** {log.get('output_length', 0)} chars\n\n")

                        prompt = log.get("prompt", "")
                        output = log.get("output", "")
                        if md_max_chars and len(prompt) > md_max_chars:
                            prompt = prompt[:md_max_chars] + "... [truncated]"
                        if md_max_chars and len(output) > md_max_chars:
                            output = output[:md_max_chars] + "... [truncated]"

                        f.write("### Prompt\n")
                        f.write(f"```\n{prompt}\n```\n\n")
                        f.write("### Output\n")
                        f.write(f"```\n{output}\n```\n\n")
                    else:
                        f.write(f"- **Tool:** `{log.get('tool', '')}`\n")
                        f.write(f"- **Parameters:** `{log.get('parameters', {})}`\n")
                        f.write(f"- **Output Length:** {log.get('output_length', 0)} chars\n\n")
                        output = log.get("output", "")
                        if md_max_chars and len(output) > md_max_chars:
                            output = output[:md_max_chars] + "... [truncated]"
                        f.write(f"### Output\n```\n{output}\n```\n\n")
                    f.write("---\n\n")
        except Exception as e:
            print(f"   Error saving tool logs: {e}")
