"""
Semantic Discovery System
------------------------
Combines news scanning with ticker semantic matching to discover
investment opportunities based on breaking news before they show up
in social media or price action.

Flow:
1. Scan news from multiple sources
2. Generate embeddings for each news item
3. Match news against ticker descriptions semantically
4. Filter and rank opportunities
5. Return actionable ticker candidates
"""

import re
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv

from tradingagents.dataflows.news_semantic_scanner import NewsSemanticScanner
from tradingagents.dataflows.ticker_semantic_db import TickerSemanticDB
from tradingagents.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class SemanticDiscovery:
    """Discovers investment opportunities through news-ticker semantic matching."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize semantic discovery system.

        Args:
            config: Configuration dict with settings for both
                    ticker DB and news scanner
        """
        self.config = config

        # Initialize ticker database
        self.ticker_db = TickerSemanticDB(config)

        # Initialize news scanner
        self.news_scanner = NewsSemanticScanner(config)

        # Discovery settings
        self.min_similarity_threshold = config.get("min_similarity_threshold", 0.3)
        self.min_news_importance = config.get("min_news_importance", 5)
        self.max_tickers_per_news = config.get("max_tickers_per_news", 5)
        self.max_total_candidates = config.get("max_total_candidates", 20)
        self.news_sentiment_filter = config.get("news_sentiment_filter", "positive")
        self.group_by_news = config.get("group_by_news", False)

    def _extract_tickers(self, mentions: List[str]) -> List[str]:
        from tradingagents.dataflows.discovery.utils import is_valid_ticker

        tickers = set()
        for mention in mentions or []:
            for match in re.findall(r"\b[A-Z]{1,5}\b", str(mention)):
                # APPLY VALIDATION IMMEDIATELY
                if is_valid_ticker(match):
                    tickers.add(match)
        return sorted(tickers)

    def get_directly_mentioned_tickers(self) -> List[Dict[str, Any]]:
        """
        Get tickers that are directly mentioned in news (highest signal).

        This extracts tickers from the 'companies_mentioned' field of news items,
        which represents explicit company references rather than semantic matches.

        Returns:
            List of ticker info dicts with news context
        """
        # Scan news if not already done
        news_items = self.news_scanner.scan_news()

        # Filter by importance
        important_news = [
            item for item in news_items if item.get("importance", 0) >= self.min_news_importance
        ]

        # Extract directly mentioned tickers
        mentioned_tickers = {}  # ticker -> list of news items

        # Common words to exclude (not tickers)
        exclude_words = {
            "A",
            "I",
            "AN",
            "AI",
            "CEO",
            "CFO",
            "CTO",
            "FDA",
            "SEC",
            "IPO",
            "ETF",
            "GDP",
            "CPI",
            "FED",
            "NYSE",
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "US",
            "UK",
            "EU",
            "AT",
            "BE",
            "BY",
            "DO",
            "GO",
            "IF",
            "IN",
            "IS",
            "IT",
            "ME",
            "MY",
            "NO",
            "OF",
            "ON",
            "OR",
            "SO",
            "TO",
            "UP",
            "WE",
            "ALL",
            "ARE",
            "FOR",
            "HAS",
            "NEW",
            "NOW",
            "OLD",
            "OUR",
            "OUT",
            "THE",
            "TOP",
            "TWO",
            "WAS",
            "WHO",
            "WHY",
            "WIN",
            "BUY",
            "COO",
            "EPS",
            "P/E",
            "ROE",
            "ROI",
            # Common business abbreviations that aren't tickers
            "INC",
            "CO",
            "LLC",
            "LTD",
            "CORP",
            "PLC",
            "AG",
            "SA",
            "SE",
            "NV",
            "GAS",
            "OIL",
            "MGE",
            "LG",  # Common words/abbreviations from logs
            # Single/two-letter words often false positives
            "AM",
            "AS",
        }

        for news_item in important_news:
            companies = news_item.get("companies_mentioned", [])
            extracted = self._extract_tickers(companies)

            for ticker in extracted:
                if ticker in exclude_words:
                    continue
                if len(ticker) < 2:
                    continue

                if ticker not in mentioned_tickers:
                    mentioned_tickers[ticker] = []

                mentioned_tickers[ticker].append(
                    {
                        "news_title": news_item.get("title", ""),
                        "news_summary": news_item.get("summary", ""),
                        "sentiment": news_item.get("sentiment", "neutral"),
                        "importance": news_item.get("importance", 5),
                        "themes": news_item.get("themes", []),
                        "source": news_item.get("source", "unknown"),
                    }
                )

        # Convert to list format, prioritizing by news importance
        result = []
        for ticker, news_list in mentioned_tickers.items():
            # Use the most important news item as primary
            best_news = max(news_list, key=lambda x: x["importance"])
            result.append(
                {
                    "ticker": ticker,
                    "news_title": best_news["news_title"],
                    "news_summary": best_news["news_summary"],
                    "sentiment": best_news["sentiment"],
                    "importance": best_news["importance"],
                    "themes": best_news["themes"],
                    "source": best_news["source"],
                    "mention_count": len(news_list),
                }
            )

        # Sort by importance and mention count
        result.sort(key=lambda x: (x["importance"], x["mention_count"]), reverse=True)

        logger.info(f"📌 Found {len(result)} directly mentioned tickers in news")

        return result[: self.max_total_candidates]

    def discover(self) -> List[Dict[str, Any]]:
        """
        Run semantic discovery to find ticker opportunities.

        Returns:
            List of ticker candidates with news context and relevance scores
        """
        logger.info("=" * 60)
        logger.info("🚀 SEMANTIC DISCOVERY")
        logger.info("=" * 60)

        # Step 1: Scan news
        news_items = self.news_scanner.scan_news()

        if not news_items:
            logger.info("No news items found.")
            return []

        # Filter news by importance threshold
        important_news = [
            item for item in news_items if item.get("importance", 0) >= self.min_news_importance
        ]

        logger.info(f"📰 Processing {len(important_news)} high-importance news items...")
        logger.info(f"(Filtered from {len(news_items)} total items)")

        if self.news_sentiment_filter:
            before_count = len(important_news)
            important_news = [
                item
                for item in important_news
                if item.get("sentiment", "").lower() == self.news_sentiment_filter
            ]
            logger.info(
                f"Sentiment filter: {self.news_sentiment_filter} "
                f"({len(important_news)}/{before_count} kept)"
            )

        # Step 2: For each news item, find matching tickers
        all_candidates = []
        news_ticker_map = {}  # Track which news items match which tickers
        news_groups = {}  # Track which tickers match each news item

        for i, news_item in enumerate(important_news, 1):
            title = news_item.get("title", "Untitled")
            logger.info(f"{i}. {title}")
            logger.debug(f"Importance: {news_item.get('importance', 0)}/10")
            mentioned_tickers = self._extract_tickers(news_item.get("companies_mentioned", []))

            # Generate search query from news
            search_text = self.news_scanner.generate_news_summary(news_item)

            # Search ticker database
            matches = self.ticker_db.search_by_text(
                query_text=search_text, top_k=self.max_tickers_per_news
            )

            # Filter by similarity threshold
            relevant_matches = [
                match
                for match in matches
                if match["similarity_score"] >= self.min_similarity_threshold
            ]

            if relevant_matches:
                logger.info(f"Found {len(relevant_matches)} relevant tickers:")
                news_key = (
                    f"{title}|{news_item.get('source', '')}|"
                    f"{news_item.get('published_at') or news_item.get('timestamp', '')}"
                )
                if news_key not in news_groups:
                    news_groups[news_key] = {
                        "news_title": title,
                        "news_summary": news_item.get("summary", ""),
                        "news_importance": news_item.get("importance", 0),
                        "news_themes": news_item.get("themes", []),
                        "news_sentiment": news_item.get("sentiment"),
                        "news_source": news_item.get("source"),
                        "published_at": news_item.get("published_at"),
                        "timestamp": news_item.get("timestamp"),
                        "mentioned_tickers": mentioned_tickers,
                        "tickers": [],
                    }
                for match in relevant_matches:
                    symbol = match["symbol"]
                    score = match["similarity_score"]
                    logger.debug(f"{symbol} (similarity: {score:.3f})")

                    # Track news-ticker mapping
                    if symbol not in news_ticker_map:
                        news_ticker_map[symbol] = []
                    news_ticker_map[symbol].append(
                        {
                            "news_title": title,
                            "news_summary": news_item.get("summary", ""),
                            "news_importance": news_item.get("importance", 0),
                            "news_themes": news_item.get("themes", []),
                            "news_sentiment": news_item.get("sentiment"),
                            "news_tickers_mentioned": mentioned_tickers,
                            "similarity_score": score,
                            "timestamp": news_item.get("timestamp"),
                            "source": news_item.get("source"),
                        }
                    )

                    if symbol not in {t["ticker"] for t in news_groups[news_key]["tickers"]}:
                        news_groups[news_key]["tickers"].append(
                            {
                                "ticker": symbol,
                                "similarity_score": score,
                                "ticker_name": match["metadata"]["name"],
                                "ticker_sector": match["metadata"]["sector"],
                                "ticker_industry": match["metadata"]["industry"],
                            }
                        )

                    # Add to candidates
                    all_candidates.append(
                        {
                            "ticker": symbol,
                            "ticker_name": match["metadata"]["name"],
                            "ticker_sector": match["metadata"]["sector"],
                            "ticker_industry": match["metadata"]["industry"],
                            "news_title": title,
                            "news_summary": news_item.get("summary", ""),
                            "news_importance": news_item.get("importance", 0),
                            "news_themes": news_item.get("themes", []),
                            "news_sentiment": news_item.get("sentiment"),
                            "news_tickers_mentioned": mentioned_tickers,
                            "similarity_score": score,
                            "news_source": news_item.get("source"),
                            "discovery_timestamp": datetime.now().isoformat(),
                        }
                    )
            else:
                logger.debug("No relevant tickers found (below threshold)")

        if self.group_by_news:
            grouped_candidates = []
            for news_entry in news_groups.values():
                tickers = news_entry["tickers"]
                if not tickers:
                    continue
                avg_similarity = sum(t["similarity_score"] for t in tickers) / len(tickers)
                aggregate_score = (
                    (news_entry["news_importance"] * 1.5)
                    + (avg_similarity * 3.0)
                    + (len(tickers) * 0.5)
                )
                grouped_candidates.append(
                    {
                        **news_entry,
                        "num_tickers": len(tickers),
                        "avg_similarity": round(avg_similarity, 3),
                        "aggregate_score": round(aggregate_score, 2),
                    }
                )

            grouped_candidates.sort(key=lambda x: x["aggregate_score"], reverse=True)
            grouped_candidates = grouped_candidates[: self.max_total_candidates]
            logger.info("📊 Aggregating and ranking news items...")
            logger.info(f"Identified {len(grouped_candidates)} news items with tickers")
            return grouped_candidates

        # Step 3: Aggregate and rank candidates
        logger.info("📊 Aggregating and ranking candidates...")

        # Group by ticker and calculate aggregate scores
        ticker_aggregates = {}
        for ticker, news_matches in news_ticker_map.items():
            # Calculate aggregate score
            # Factors: number of news matches, importance, similarity
            num_matches = len(news_matches)
            avg_importance = sum(n["news_importance"] for n in news_matches) / num_matches
            avg_similarity = sum(n["similarity_score"] for n in news_matches) / num_matches
            max_importance = max(n["news_importance"] for n in news_matches)

            # Weighted score
            aggregate_score = (
                (num_matches * 2.0)  # More news = higher score
                + (avg_importance * 1.5)  # Average importance
                + (avg_similarity * 3.0)  # Similarity strength
                + (max_importance * 1.0)  # Bonus for having one very important match
            )

            ticker_aggregates[ticker] = {
                "ticker": ticker,
                "num_news_matches": num_matches,
                "avg_importance": round(avg_importance, 2),
                "avg_similarity": round(avg_similarity, 3),
                "max_importance": max_importance,
                "aggregate_score": round(aggregate_score, 2),
                "news_matches": news_matches,
            }

        # Sort by aggregate score
        ranked_candidates = sorted(
            ticker_aggregates.values(), key=lambda x: x["aggregate_score"], reverse=True
        )

        # Limit to max candidates
        ranked_candidates = ranked_candidates[: self.max_total_candidates]

        logger.info(f"Identified {len(ranked_candidates)} unique ticker candidates")

        return ranked_candidates

    def format_discovery_report(self, candidates: List[Dict[str, Any]]) -> str:
        """
        Format discovery results as a readable report.

        Args:
            candidates: List of ranked candidates

        Returns:
            Formatted text report
        """
        if not candidates:
            return "No opportunities discovered."

        if "tickers" in candidates[0]:
            report = "\n" + "=" * 60
            report += "\n📰 NEWS-DRIVEN RESULTS"
            report += "\n" + "=" * 60 + "\n"

            for i, news in enumerate(candidates, 1):
                title = news["news_title"]
                score = news["aggregate_score"]
                num_tickers = news["num_tickers"]
                importance = news["news_importance"]

                report += f"\n{i}. {title}"
                report += f"\n   Score: {score:.2f} | Tickers: {num_tickers} | Importance: {importance}/10"
                report += f"\n   Source: {news.get('news_source', 'unknown')}"
                if news.get("news_themes"):
                    report += f"\n   Themes: {', '.join(news['news_themes'])}"
                if news.get("news_summary"):
                    report += f"\n   Summary: {news['news_summary']}"
                if news.get("mentioned_tickers"):
                    report += f"\n   Mentioned Tickers: {', '.join(news['mentioned_tickers'])}"

                tickers = sorted(news["tickers"], key=lambda x: x["similarity_score"], reverse=True)
                report += "\n   Related Tickers:"
                for j, ticker_info in enumerate(tickers[:5], 1):
                    report += (
                        f"\n      {j}. {ticker_info['ticker']} "
                        f"(similarity: {ticker_info['similarity_score']:.3f})"
                    )

                if len(tickers) > 5:
                    report += f"\n      ... and {len(tickers) - 5} more"

                report += "\n"

            return report

        report = "\n" + "=" * 60
        report += "\n🎯 SEMANTIC DISCOVERY RESULTS"
        report += "\n" + "=" * 60 + "\n"

        for i, candidate in enumerate(candidates, 1):
            ticker = candidate["ticker"]
            score = candidate["aggregate_score"]
            num_matches = candidate["num_news_matches"]
            avg_importance = candidate["avg_importance"]

            report += f"\n{i}. {ticker}"
            report += f"\n   Score: {score:.2f} | Matches: {num_matches} | Avg Importance: {avg_importance}/10"
            report += "\n   Related News:"

            for j, news in enumerate(candidate["news_matches"][:3], 1):  # Show top 3 news
                report += f"\n      {j}. {news['news_title']}"
                report += f"\n         Similarity: {news['similarity_score']:.3f} | Importance: {news['news_importance']}/10"
                if news.get("news_themes"):
                    report += f"\n         Themes: {', '.join(news['news_themes'])}"

            if len(candidate["news_matches"]) > 3:
                report += f"\n      ... and {len(candidate['news_matches']) - 3} more"

            report += "\n"

        return report


def main():
    """CLI for running semantic discovery."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run semantic discovery")
    parser.add_argument(
        "--news-sources",
        nargs="+",
        default=["openai"],
        choices=["openai", "google_news", "sec_filings", "alpha_vantage", "gemini_search"],
        help="News sources to use",
    )
    parser.add_argument(
        "--min-importance", type=int, default=5, help="Minimum news importance (1-10)"
    )
    parser.add_argument(
        "--min-similarity", type=float, default=0.2, help="Minimum similarity threshold (0-1)"
    )
    parser.add_argument(
        "--max-candidates", type=int, default=15, help="Maximum ticker candidates to return"
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="How far back to look for news (in hours). Examples: 1, 6, 24, 168",
    )
    parser.add_argument("--output", type=str, help="Output file for results JSON")
    parser.add_argument(
        "--group-by-news", action="store_true", help="Group results by news item instead of ticker"
    )

    args = parser.parse_args()

    # Load project config
    from tradingagents.default_config import DEFAULT_CONFIG

    config = {
        "project_dir": DEFAULT_CONFIG["project_dir"],
        "use_openai_embeddings": True,
        "news_sources": args.news_sources,
        "news_lookback_hours": args.lookback_hours,
        "min_news_importance": args.min_importance,
        "min_similarity_threshold": args.min_similarity,
        "max_tickers_per_news": 5,
        "max_total_candidates": args.max_candidates,
        "news_sentiment_filter": "positive",
        "group_by_news": args.group_by_news,
    }

    # Run discovery
    discovery = SemanticDiscovery(config)
    candidates = discovery.discover()

    # Display report
    report = discovery.format_discovery_report(candidates)
    logger.info(report)

    # Save to file if specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(candidates, f, indent=2)
        logger.info(f"✅ Saved {len(candidates)} candidates to {args.output}")


if __name__ == "__main__":
    main()
