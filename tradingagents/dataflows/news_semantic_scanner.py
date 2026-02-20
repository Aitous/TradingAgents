"""
News Semantic Scanner
--------------------
Scans news from multiple sources, summarizes key themes, and enables semantic
matching against ticker descriptions to find relevant investment opportunities.

Sources:
- OpenAI web search (real-time market news)
- SEC EDGAR filings (regulatory news)
- Google News
- Alpha Vantage news
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI

from tradingagents.dataflows.discovery.utils import build_llm_log_entry
from tradingagents.schemas import FilingsList, NewsList
from tradingagents.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class NewsSemanticScanner:
    """Scans and processes news for semantic ticker matching."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize news scanner.

        Args:
            config: Configuration dict with:
                - openai_api_key: OpenAI API key
                - news_sources: List of sources to use
                - max_news_items: Maximum news items to process
                - news_lookback_hours: How far back to look for news (default: 24 hours)
        """
        self.config = config
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.news_sources = config.get("news_sources", ["openai", "google_news"])
        self.max_news_items = config.get("max_news_items", 20)
        self.news_lookback_hours = config.get("news_lookback_hours", 24)
        self.log_callback = config.get("log_callback")

        # Calculate time window
        self.cutoff_time = datetime.now() - timedelta(hours=self.news_lookback_hours)

    def _emit_log(self, entry: Dict[str, Any]) -> None:
        if self.log_callback:
            try:
                self.log_callback(entry)
            except Exception:
                pass

    def _log_llm(
        self,
        step: str,
        model: str,
        prompt: Any,
        output: Any,
        error: str = "",
    ) -> None:
        entry = build_llm_log_entry(
            node="semantic_news",
            step=step,
            model=model,
            prompt=prompt,
            output=output,
            error=error,
        )
        self._emit_log(entry)

    def _get_time_phrase(self) -> str:
        """Generate human-readable time phrase for queries."""
        if self.news_lookback_hours <= 1:
            return "from the last hour"
        elif self.news_lookback_hours <= 6:
            return f"from the last {self.news_lookback_hours} hours"
        elif self.news_lookback_hours <= 24:
            return "from today"
        elif self.news_lookback_hours <= 48:
            return "from the last 2 days"
        else:
            days = int(self.news_lookback_hours / 24)
            return f"from the last {days} days"

    def _deduplicate_news(
        self, news_items: List[Dict[str, Any]], similarity_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate news items using semantic similarity (embeddings + cosine similarity).

        Two-pass approach:
        1. Fast hash-based pass for exact/near-exact duplicates
        2. Embedding-based cosine similarity for semantically similar stories

        Args:
            news_items: List of news items from various sources
            similarity_threshold: Cosine similarity threshold (0.85 = very similar)

        Returns:
            Deduplicated list, keeping highest importance version of each story
        """
        import hashlib
        import re

        import numpy as np

        if not news_items:
            return []

        def normalize_text(text: str) -> str:
            """Normalize text for comparison."""
            if not text:
                return ""
            text = text.lower()
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        def get_content_hash(item: Dict[str, Any]) -> str:
            """Generate hash from normalized title + summary."""
            title = normalize_text(item.get("title", ""))
            summary = normalize_text(item.get("summary", ""))[:100]
            content = title + " " + summary
            return hashlib.md5(content.encode()).hexdigest()

        def get_news_text(item: Dict[str, Any]) -> str:
            """Get combined text for embedding."""
            title = item.get("title", "")
            summary = item.get("summary", "")
            return f"{title}. {summary}"[:500]  # Limit length for efficiency

        def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
            """Compute cosine similarity between two vectors."""
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        # === PASS 1: Hash-based deduplication (fast, exact matches) ===
        seen_hashes: Dict[str, Dict[str, Any]] = {}
        hash_duplicates = 0

        for item in news_items:
            content_hash = get_content_hash(item)
            if content_hash not in seen_hashes:
                seen_hashes[content_hash] = item
            else:
                existing = seen_hashes[content_hash]
                if (item.get("importance", 0) or 0) > (existing.get("importance", 0) or 0):
                    seen_hashes[content_hash] = item
                hash_duplicates += 1

        after_hash = list(seen_hashes.values())
        logger.info(
            f"Hash dedup: {len(news_items)} → {len(after_hash)} ({hash_duplicates} exact duplicates)"
        )

        # === PASS 2: Embedding-based semantic similarity ===
        # Only run if we have enough items to justify the cost
        if len(after_hash) <= 3:
            return after_hash

        try:
            # Generate embeddings for all remaining items
            texts = [get_news_text(item) for item in after_hash]

            # Use OpenAI embeddings (same as ticker_semantic_db)
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            embeddings = np.array([e.embedding for e in response.data])

            # Find semantic duplicates using cosine similarity
            unique_indices = []
            semantic_duplicates = 0

            for i in range(len(after_hash)):
                is_duplicate = False

                for j in unique_indices:
                    sim = cosine_similarity(embeddings[i], embeddings[j])
                    if sim >= similarity_threshold:
                        # This is a semantic duplicate
                        is_duplicate = True
                        semantic_duplicates += 1

                        # Keep higher importance version
                        existing_item = after_hash[j]
                        new_item = after_hash[i]
                        if (new_item.get("importance", 0) or 0) > (
                            existing_item.get("importance", 0) or 0
                        ):
                            # Replace with higher importance
                            unique_indices.remove(j)
                            unique_indices.append(i)

                        logger.debug(
                            f"Semantic duplicate (sim={sim:.2f}): "
                            f"'{new_item.get('title', '')[:40]}' vs "
                            f"'{existing_item.get('title', '')[:40]}'"
                        )
                        break

                if not is_duplicate:
                    unique_indices.append(i)

            final_items = [after_hash[i] for i in unique_indices]
            logger.info(
                f"Semantic dedup: {len(after_hash)} → {len(final_items)} "
                f"({semantic_duplicates} similar stories merged)"
            )

            return final_items

        except Exception as e:
            logger.warning(f"Embedding-based dedup failed, using hash-only results: {e}")
            return after_hash

    def _filter_by_time(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter news items by timestamp to respect lookback window.

        Args:
            news_items: List of news items with 'published_at' or 'timestamp' field

        Returns:
            Filtered list of news items within time window
        """
        filtered = []
        filtered_out_count = 0

        for item in news_items:
            timestamp_str = item.get("published_at") or item.get("timestamp")
            title_preview = item.get("title", "")[:60]

            if not timestamp_str:
                # No timestamp, keep it (assume recent)
                logger.debug(f"No timestamp for '{title_preview}', keeping")
                filtered.append(item)
                continue

            item_time = self._parse_timestamp(timestamp_str, date_only_end=True)
            if not item_time:
                # If parsing fails, keep it
                logger.debug(f"Parse failed for '{timestamp_str}' on '{title_preview}', keeping")
                filtered.append(item)
                continue

            if item_time >= self.cutoff_time:
                filtered.append(item)
            else:
                filtered_out_count += 1
                logger.debug(
                    f"FILTERED OUT: '{title_preview}' | "
                    f"published_at='{item.get('published_at')}' | "
                    f"parsed={item_time.strftime('%Y-%m-%d %H:%M')} | "
                    f"cutoff={self.cutoff_time.strftime('%Y-%m-%d %H:%M')}"
                )

        if filtered_out_count > 0:
            logger.info(
                f"Time filter removed {filtered_out_count} items with timestamps before cutoff"
            )

        return filtered

    def _parse_timestamp(self, timestamp_str: str, date_only_end: bool) -> Optional[datetime]:
        """Parse a timestamp string into a naive datetime, or return None if invalid."""
        try:
            # Handle date-only strings
            if len(timestamp_str) == 10 and timestamp_str[4] == "-" and timestamp_str[7] == "-":
                base_time = datetime.fromisoformat(timestamp_str)
                if date_only_end:
                    return base_time.replace(hour=23, minute=59, second=59)
                return base_time

            # Parse ISO timestamp
            parsed_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if parsed_time.tzinfo:
                parsed_time = parsed_time.astimezone().replace(tzinfo=None)
            return parsed_time
        except Exception:
            return None

    def _publish_date_range(
        self, news_items: List[Dict[str, Any]]
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the earliest and latest publish timestamps from a list of news items."""
        min_time = None
        max_time = None
        for item in news_items:
            timestamp_str = item.get("published_at") or item.get("timestamp")
            if not timestamp_str:
                continue
            item_time = self._parse_timestamp(timestamp_str, date_only_end=False)
            if not item_time:
                continue
            if min_time is None or item_time < min_time:
                min_time = item_time
            if max_time is None or item_time > max_time:
                max_time = item_time
        return min_time, max_time

    def _build_time_constraint(self) -> str:
        """Build the shared time constraint block used by all news prompts."""
        current_datetime = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        cutoff_datetime = self.cutoff_time.strftime("%Y-%m-%dT%H:%M:%S")
        return (
            f"CRITICAL TIME CONSTRAINT:\n"
            f"- Current time: {current_datetime}\n"
            f"- Only include items published AFTER: {cutoff_datetime}\n"
            f"- Skip anything older than {self.news_lookback_hours} hours"
        )

    def _build_extraction_fields(self, detail_level: str = "full") -> str:
        """Build the shared extraction fields block.

        Args:
            detail_level: "full" for primary searches, "brief" for parsing raw feeds.
        """
        current_datetime = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        base = "For each item, extract:\n" "- title: Headline\n"
        if detail_level == "full":
            base += "- summary: 2-3 sentence summary of key points\n"
        else:
            base += "- summary: Brief summary of key points\n"
        base += (
            f"- published_at: ISO-8601 timestamp (REQUIRED — convert relative times like '2 hours ago' to full timestamp using current time {current_datetime})\n"
            "- companies_mentioned: List of stock ticker symbols (prefer tickers over company names, e.g. 'AAPL' not 'Apple Inc.')\n"
            "- themes: Key themes (e.g., 'earnings beat', 'FDA approval', 'merger', 'insider buying')\n"
            "- sentiment: one of positive, negative, neutral\n"
            "- importance: 1-10 score (10 = highly market-moving, company-specific catalysts score higher than broad market news)"
        )
        return base

    _COMPANY_SPECIFIC_INSTRUCTION = (
        "Prefer company-specific or single-catalyst stories that impact one company or a small "
        "group of companies. Avoid broad market, index, or macroeconomic headlines unless they "
        "have a clear company-specific catalyst. If a story is sector-wide without a specific "
        "company catalyst, skip it."
    )

    def _build_web_search_prompt(self, query: str = "breaking stock market news today") -> str:
        """
        Build unified web search prompt for both OpenAI and Gemini.

        Args:
            query: Search query for news

        Returns:
            Formatted search prompt string
        """
        time_phrase = self._get_time_phrase()
        time_query = f"{query} {time_phrase}"

        return f"""Search the web for: {time_query}

{self._build_time_constraint()}

Find the top {self.max_news_items} most important market-moving news stories from the last {self.news_lookback_hours} hours.

{self._COMPANY_SPECIFIC_INSTRUCTION}

Focus on:
- Earnings reports and guidance
- FDA approvals / regulatory decisions
- Mergers, acquisitions, partnerships
- Product launches
- Executive changes
- Legal/regulatory actions
- Analyst upgrades/downgrades

{self._build_extraction_fields("full")}
"""

    def _build_openai_input(self, system_text: str, user_text: str) -> str:
        """Build Responses API input as a single prompt string."""
        if system_text:
            return f"{system_text}\n\n{user_text}"
        return user_text

    def _fetch_openai_news(
        self, query: str = "breaking stock market news today"
    ) -> List[Dict[str, Any]]:
        """
        Fetch news using OpenAI's web search capability.

        Args:
            query: Search query for news

        Returns:
            List of news items with title, summary, published_at, timestamp
        """
        try:
            # Build search prompt
            search_prompt = self._build_web_search_prompt(query)

            # Use OpenAI web search tool for real-time news
            response = self.openai_client.responses.parse(
                model="gpt-4o",
                tools=[{"type": "web_search"}],
                input=self._build_openai_input(
                    "You are a financial news analyst. Search the web for the latest market news "
                    "and return structured summaries.",
                    search_prompt,
                ),
                text_format=NewsList,
            )

            news_list = response.output_parsed
            news_items = [item.model_dump() for item in news_list.news]

            self._log_llm(
                step="OpenAI web search",
                model="gpt-4o",
                prompt=search_prompt,
                output=news_items,
            )

            # Add metadata
            for item in news_items:
                item["source"] = "openai_search"
                item["timestamp"] = datetime.now().isoformat()

            return news_items[: self.max_news_items]

        except Exception as e:
            self._log_llm(
                step="OpenAI web search",
                model="gpt-4o",
                prompt=search_prompt if "search_prompt" in locals() else "",
                output="",
                error=str(e),
            )
            logger.error(f"Error fetching OpenAI news: {e}")
            return []

    def _fetch_google_news(self, query: str = "stock market") -> List[Dict[str, Any]]:
        """
        Fetch news from Google News RSS.

        Args:
            query: Search query

        Returns:
            List of news items
        """
        try:
            # Use Google News helper
            from tradingagents.dataflows.google import get_google_news

            # Convert hours to days (round up)
            lookback_days = max(1, int((self.news_lookback_hours + 23) / 24))

            news_report = get_google_news(
                query=query,
                curr_date=datetime.now().strftime("%Y-%m-%d"),
                look_back_days=lookback_days,
            )

            # Parse the report using LLM to extract structured data
            parse_prompt = f"""Parse this news report and extract individual news items.

{self._build_time_constraint()}

{self._COMPANY_SPECIFIC_INSTRUCTION}

{news_report}

{self._build_extraction_fields("brief")}

Return as JSON array with key "news"."""
            response = self.openai_client.responses.parse(
                model="gpt-4o-mini",
                input=self._build_openai_input(
                    "Extract news items from this report into structured JSON format.",
                    parse_prompt,
                ),
                text_format=NewsList,
            )

            news_list = response.output_parsed
            news_items = [item.model_dump() for item in news_list.news]

            self._log_llm(
                step="Parse Google News",
                model="gpt-4o-mini",
                prompt=parse_prompt,
                output=news_items,
            )

            # Add metadata
            for item in news_items:
                item["source"] = "google_news"
                item["timestamp"] = datetime.now().isoformat()

            return news_items[: self.max_news_items]

        except Exception as e:
            self._log_llm(
                step="Parse Google News",
                model="gpt-4o-mini",
                prompt=parse_prompt if "parse_prompt" in locals() else "",
                output="",
                error=str(e),
            )
            logger.error(f"Error fetching Google News: {e}")
            return []

    def _fetch_sec_filings(self) -> List[Dict[str, Any]]:
        """
        Fetch recent SEC filings (8-K, 13D, 13G - market-moving events).

        Returns:
            List of filing summaries
        """
        try:
            # SEC EDGAR API endpoint
            # Get recent 8-K filings (material events)
            url = "https://www.sec.gov/cgi-bin/browse-edgar"
            params = {"action": "getcurrent", "type": "8-K", "output": "atom", "count": 20}
            headers = {"User-Agent": "TradingAgents/1.0 (contact@example.com)"}

            response = requests.get(url, params=params, headers=headers, timeout=10)

            if response.status_code != 200:
                return []

            # Parse SEC filings using LLM
            # (SEC returns XML/Atom feed, we'll parse with LLM for simplicity)
            filings_prompt = f"""Parse these SEC 8-K filings and extract the most important material events.

{self._build_time_constraint()}

Prefer company-specific filings and material events; skip broad market commentary or routine filings.

{response.text}

{self._build_extraction_fields("brief")}

Return as JSON array with key "filings"."""
            llm_response = self.openai_client.responses.parse(
                model="gpt-4o-mini",
                input=self._build_openai_input(
                    "Extract important SEC 8-K filings from this data and summarize the market-moving events.",
                    filings_prompt,
                ),
                text_format=FilingsList,
            )

            filings_list = llm_response.output_parsed
            filings = [item.model_dump() for item in filings_list.filings]

            self._log_llm(
                step="Parse SEC filings",
                model="gpt-4o-mini",
                prompt=filings_prompt,
                output=filings,
            )

            # Add metadata
            for filing in filings:
                filing["source"] = "sec_edgar"
                filing["timestamp"] = datetime.now().isoformat()

            return filings[: self.max_news_items]

        except Exception as e:
            self._log_llm(
                step="Parse SEC filings",
                model="gpt-4o-mini",
                prompt=filings_prompt if "filings_prompt" in locals() else "",
                output="",
                error=str(e),
            )
            logger.error(f"Error fetching SEC filings: {e}")
            return []

    def _fetch_alpha_vantage_news(
        self, topics: str = "earnings,technology"
    ) -> List[Dict[str, Any]]:
        """
        Fetch news from Alpha Vantage.

        Args:
            topics: News topics to filter

        Returns:
            List of news items
        """
        try:
            from tradingagents.dataflows.alpha_vantage_news import get_alpha_vantage_news_feed

            # Use cutoff time for Alpha Vantage
            time_from = self.cutoff_time.strftime("%Y%m%dT%H%M")

            news_report = get_alpha_vantage_news_feed(topics=topics, time_from=time_from, limit=50)

            # Parse with LLM
            parse_prompt = f"""Parse this news feed and extract the most important market-moving stories.

{self._build_time_constraint()}

{self._COMPANY_SPECIFIC_INSTRUCTION}

{news_report}

{self._build_extraction_fields("brief")}

Return as JSON array with key "news"."""
            response = self.openai_client.responses.parse(
                model="gpt-4o-mini",
                input=self._build_openai_input(
                    "Extract and summarize important market news.",
                    parse_prompt,
                ),
                text_format=NewsList,
            )

            news_list = response.output_parsed
            news_items = [item.model_dump() for item in news_list.news]

            self._log_llm(
                step="Parse Alpha Vantage news",
                model="gpt-4o-mini",
                prompt=parse_prompt,
                output=news_items,
            )

            # Add metadata
            for item in news_items:
                item["source"] = "alpha_vantage"
                item["timestamp"] = datetime.now().isoformat()

            return news_items[: self.max_news_items]

        except Exception as e:
            self._log_llm(
                step="Parse Alpha Vantage news",
                model="gpt-4o-mini",
                prompt=parse_prompt if "parse_prompt" in locals() else "",
                output="",
                error=str(e),
            )
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []

    def _fetch_gemini_search_news(
        self, query: str = "breaking stock market news today"
    ) -> List[Dict[str, Any]]:
        """
        Fetch news using Google Gemini's native web search (grounding) capability.

        This uses Gemini's built-in web search tool for real-time market news,
        which may provide different results than OpenAI's web search.

        Args:
            query: Search query for news

        Returns:
            List of news items with title, summary, published_at, timestamp
        """
        try:
            import os

            # Get API key
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                logger.error("GOOGLE_API_KEY not set, skipping Gemini search")
                return []

            # Build search prompt
            search_prompt = self._build_web_search_prompt(query)

            # Step 1: Execute web search using Gemini with google_search tool
            search_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",  # Fast model for search
                api_key=google_api_key,
                temperature=1.0,  # Higher temperature for diverse results
            ).bind_tools([{"google_search": {}}])

            # Execute search
            raw_response = search_llm.invoke(search_prompt)
            self._log_llm(
                step="Gemini search",
                model="gemini-2.5-flash-lite",
                prompt=search_prompt,
                output=raw_response.content if hasattr(raw_response, "content") else raw_response,
            )

            # Step 2: Structure the results using Gemini with JSON schema
            structured_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite", api_key=google_api_key
            ).with_structured_output(NewsList, method="json_schema")

            structure_prompt = f"""Parse the following web search results into structured news items.

{self._build_time_constraint()}

{self._build_extraction_fields("full")}

Web search results:
{raw_response.content}

Return as JSON with "news" array."""

            structured_response = structured_llm.invoke(structure_prompt)
            self._log_llm(
                step="Gemini search structuring",
                model="gemini-2.5-flash-lite",
                prompt=structure_prompt,
                output=structured_response,
            )

            # Extract news items
            news_items = [item.model_dump() for item in structured_response.news]

            # Add metadata
            for item in news_items:
                item["source"] = "gemini_search"
                item["timestamp"] = datetime.now().isoformat()

            return news_items[: self.max_news_items]

        except Exception as e:
            self._log_llm(
                step="Gemini search",
                model="gemini-2.5-flash-lite",
                prompt=search_prompt if "search_prompt" in locals() else "",
                output="",
                error=str(e),
            )
            logger.error(f"Error fetching Gemini search news: {e}")
            return []

    def scan_news(self) -> List[Dict[str, Any]]:
        """
        Scan news from all enabled sources.

        Returns:
            Aggregated list of news items sorted by importance
        """
        all_news = []

        logger.info("Scanning news sources...")
        logger.info(f"Time window: {self._get_time_phrase()} (last {self.news_lookback_hours}h)")
        logger.info(f"Cutoff: {self.cutoff_time.strftime('%Y-%m-%d %H:%M')}")

        # Fetch from each enabled source
        if "openai" in self.news_sources:
            logger.info("Fetching OpenAI web search...")
            openai_news = self._fetch_openai_news()
            all_news.extend(openai_news)
            logger.info(f"Found {len(openai_news)} items from OpenAI")
            min_date, max_date = self._publish_date_range(openai_news)
            if min_date:
                logger.debug(f"Min publish date (OpenAI): {min_date.strftime('%Y-%m-%d %H:%M')}")
            else:
                logger.debug("Min publish date (OpenAI): N/A")
            if max_date:
                logger.debug(f"Max publish date (OpenAI): {max_date.strftime('%Y-%m-%d %H:%M')}")
            else:
                logger.debug("Max publish date (OpenAI): N/A")

        if "google_news" in self.news_sources:
            logger.info("Fetching Google News...")
            google_news = self._fetch_google_news()
            all_news.extend(google_news)
            logger.info(f"Found {len(google_news)} items from Google News")
            min_date, max_date = self._publish_date_range(google_news)
            if min_date:
                logger.debug(
                    f"Min publish date (Google News): {min_date.strftime('%Y-%m-%d %H:%M')}"
                )
            else:
                logger.debug("Min publish date (Google News): N/A")
            if max_date:
                logger.debug(
                    f"Max publish date (Google News): {max_date.strftime('%Y-%m-%d %H:%M')}"
                )
            else:
                logger.debug("Max publish date (Google News): N/A")

        if "sec_filings" in self.news_sources:
            logger.info("Fetching SEC filings...")
            sec_filings = self._fetch_sec_filings()
            all_news.extend(sec_filings)
            logger.info(f"Found {len(sec_filings)} items from SEC")
            min_date, max_date = self._publish_date_range(sec_filings)
            if min_date:
                logger.debug(f"Min publish date (SEC): {min_date.strftime('%Y-%m-%d %H:%M')}")
            else:
                logger.debug("Min publish date (SEC): N/A")
            if max_date:
                logger.debug(f"Max publish date (SEC): {max_date.strftime('%Y-%m-%d %H:%M')}")
            else:
                logger.debug("Max publish date (SEC): N/A")

        if "alpha_vantage" in self.news_sources:
            logger.info("Fetching Alpha Vantage news...")
            av_news = self._fetch_alpha_vantage_news()
            all_news.extend(av_news)
            logger.info(f"Found {len(av_news)} items from Alpha Vantage")
            min_date, max_date = self._publish_date_range(av_news)
            if min_date:
                logger.debug(
                    f"Min publish date (Alpha Vantage): {min_date.strftime('%Y-%m-%d %H:%M')}"
                )
            else:
                logger.debug("Min publish date (Alpha Vantage): N/A")
            if max_date:
                logger.debug(
                    f"Max publish date (Alpha Vantage): {max_date.strftime('%Y-%m-%d %H:%M')}"
                )
            else:
                logger.debug("Max publish date (Alpha Vantage): N/A")

        if "gemini_search" in self.news_sources:
            logger.info("Fetching Google Gemini search...")
            gemini_news = self._fetch_gemini_search_news()
            all_news.extend(gemini_news)
            logger.info(f"Found {len(gemini_news)} items from Gemini search")
            min_date, max_date = self._publish_date_range(gemini_news)
            if min_date:
                logger.debug(f"Min publish date (Gemini): {min_date.strftime('%Y-%m-%d %H:%M')}")
            else:
                logger.debug("Min publish date (Gemini): N/A")
            if max_date:
                logger.debug(f"Max publish date (Gemini): {max_date.strftime('%Y-%m-%d %H:%M')}")
            else:
                logger.debug("Max publish date (Gemini): N/A")

        # Apply time filtering
        logger.info(f"Collected {len(all_news)} raw news items")
        all_news = self._filter_by_time(all_news)
        logger.info(f"After time filtering: {len(all_news)} items")

        # Deduplicate news from multiple sources (same story = same hash)
        all_news = self._deduplicate_news(all_news)
        logger.info(f"After deduplication: {len(all_news)} items")

        # Sort by importance
        all_news.sort(key=lambda x: x.get("importance", 0), reverse=True)

        logger.info(f"Total news items collected: {len(all_news)}")

        return all_news[: self.max_news_items]

    def generate_news_summary(self, news_item: Dict[str, Any]) -> str:
        """
        Generate a semantic search-optimized summary for a news item.

        Args:
            news_item: News item dict

        Returns:
            Optimized summary text for embedding/matching
        """
        title = news_item.get("title", "")
        summary = news_item.get("summary", "")
        themes = news_item.get("themes", [])
        companies = news_item.get("companies_mentioned", [])

        # Create rich text for semantic matching
        search_text = f"""
        {title}

        {summary}

        Key themes: {', '.join(themes) if themes else 'General market news'}
        Companies mentioned: {', '.join(companies) if companies else 'Broad market'}
        """.strip()

        return search_text


def main():
    """CLI for testing news scanner."""
    import argparse

    parser = argparse.ArgumentParser(description="Scan news for semantic ticker matching")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["openai"],
        choices=["openai", "google_news", "sec_filings", "alpha_vantage", "gemini_search"],
        help="News sources to use",
    )
    parser.add_argument("--max-items", type=int, default=10, help="Maximum news items to fetch")
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="How far back to look for news (in hours). Examples: 1 (last hour), 6 (last 6 hours), 24 (last day), 168 (last week)",
    )
    parser.add_argument("--output", type=str, help="Output file for news JSON")

    args = parser.parse_args()

    config = {
        "news_sources": args.sources,
        "max_news_items": args.max_items,
        "news_lookback_hours": args.lookback_hours,
    }

    scanner = NewsSemanticScanner(config)
    news_items = scanner.scan_news()

    # Display results
    logger.info("\n" + "=" * 60)
    logger.info(f"Top {min(5, len(news_items))} Most Important News Items:")
    logger.info("=" * 60 + "\n")

    for i, item in enumerate(news_items[:5], 1):
        logger.info(f"{i}. {item.get('title', 'Untitled')}")
        logger.info(f"   Source: {item.get('source', 'unknown')}")
        logger.info(f"   Importance: {item.get('importance', 'N/A')}/10")
        logger.info(f"   Summary: {item.get('summary', '')[:150]}...")
        logger.info(f"   Themes: {', '.join(item.get('themes', []))}")
        logger.info("")

    # Save to file if specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(news_items, f, indent=2)
        logger.info(f"✅ Saved {len(news_items)} news items to {args.output}")


if __name__ == "__main__":
    main()
