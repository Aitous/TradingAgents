from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, StateGraph

from tradingagents.agents.utils.agent_states import DiscoveryState
from tradingagents.dataflows.discovery import scanners  # Load scanners to trigger registration
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY
from tradingagents.dataflows.discovery.utils import PRIORITY_ORDER, Priority, serialize_for_log
from tradingagents.tools.executor import execute_tool


# Known PERMANENTLY delisted tickers (verified mergers, bankruptcies, delistings)
# NOTE: This list should only contain tickers that are CONFIRMED to be permanently delisted.
# Do NOT add actively traded stocks here. Use the dynamic delisted_cache for uncertain cases.
def get_delisted_tickers():
    """Get combined list of delisted tickers from permanent list + dynamic cache."""
    from tradingagents.dataflows.discovery.utils import get_delisted_tickers

    return get_delisted_tickers()


def is_valid_ticker(ticker: str) -> bool:
    """Validate if a ticker is tradeable and not junk."""
    from tradingagents.dataflows.discovery.utils import is_valid_ticker

    return is_valid_ticker(ticker)


class DiscoveryGraph:
    """
    Discovery Graph for finding investment opportunities.

    Orchestrates the discovery workflow: scanning -> filtering -> ranking.
    Supports traditional, semantic, and hybrid discovery modes.
    """

    # Node names
    NODE_SCANNER = "scanner"
    NODE_FILTER = "filter"
    NODE_RANKER = "ranker"

    # Source types
    SOURCE_NEWS_MENTION = "news_direct_mention"
    SOURCE_SEMANTIC = "semantic_news_match"
    SOURCE_UNKNOWN = "unknown"

    # Priority levels (lower number = higher priority)
    PRIORITY_ORDER = PRIORITY_ORDER

    # Priority level names
    PRIORITY_CRITICAL = Priority.CRITICAL.value
    PRIORITY_HIGH = Priority.HIGH.value
    PRIORITY_MEDIUM = Priority.MEDIUM.value
    PRIORITY_LOW = Priority.LOW.value
    PRIORITY_UNKNOWN = Priority.UNKNOWN.value

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Discovery Graph.

        Args:
            config: Configuration dictionary containing:
                - llm_provider: LLM provider (e.g., 'openai', 'google')
                - discovery: Discovery-specific settings
                - results_dir: Directory for saving results
        """
        self.config = config or {}

        # Initialize LLMs
        from tradingagents.utils.llm_factory import create_llms

        self.deep_thinking_llm, self.quick_thinking_llm = create_llms(self.config)

        # Load configurations
        self._load_discovery_config()
        self._load_logging_config()

        # Store run directory for saving results
        self.run_dir = self.config.get("discovery_run_dir", None)

        # Initialize Analytics
        from tradingagents.dataflows.discovery.analytics import DiscoveryAnalytics

        self.analytics = DiscoveryAnalytics(data_dir="data")

        self.graph = self._create_graph()

    def _load_discovery_config(self) -> None:
        """Load discovery-specific configuration with defaults."""
        discovery_config = self.config.get("discovery", {})

        # Scanner limits
        self.reddit_trending_limit = discovery_config.get("reddit_trending_limit", 15)
        self.market_movers_limit = discovery_config.get("market_movers_limit", 10)
        self.max_candidates_to_analyze = discovery_config.get("max_candidates_to_analyze", 100)
        self.analyze_all_candidates = discovery_config.get("analyze_all_candidates", False)
        self.final_recommendations = discovery_config.get("final_recommendations", 3)
        self.deep_dive_max_workers = discovery_config.get("deep_dive_max_workers", 3)

        # Volume and movement filters
        self.min_average_volume = discovery_config.get("min_average_volume", 0)
        self.volume_lookback_days = discovery_config.get("volume_lookback_days", 20)
        self.volume_cache_key = discovery_config.get("volume_cache_key", "default")
        self.filter_same_day_movers = discovery_config.get("filter_same_day_movers", True)
        self.intraday_movement_threshold = discovery_config.get("intraday_movement_threshold", 15.0)

        # Earnings discovery limits
        self.max_earnings_candidates = discovery_config.get("max_earnings_candidates", 50)
        self.max_days_until_earnings = discovery_config.get("max_days_until_earnings", 7)
        self.min_market_cap = discovery_config.get(
            "min_market_cap", 0
        )  # In billions, 0 = no filter

        # News settings
        self.news_lookback_days = discovery_config.get("news_lookback_days", 7)
        self.batch_news_vendor = discovery_config.get("batch_news_vendor", "openai")
        self.batch_news_batch_size = discovery_config.get("batch_news_batch_size", 50)

        # Discovery mode: "traditional", "semantic", or "hybrid"
        self.discovery_mode = discovery_config.get("discovery_mode", "hybrid")

        # Semantic discovery settings
        self.semantic_news_sources = discovery_config.get("semantic_news_sources", ["openai"])
        self.semantic_news_lookback_hours = discovery_config.get("semantic_news_lookback_hours", 24)
        self.semantic_min_news_importance = discovery_config.get("semantic_min_news_importance", 5)
        self.semantic_min_similarity = discovery_config.get("semantic_min_similarity", 0.2)
        self.semantic_max_tickers_per_news = discovery_config.get(
            "semantic_max_tickers_per_news", 5
        )

        # Console price charts
        self.console_price_charts = discovery_config.get("console_price_charts", False)
        self.price_chart_library = discovery_config.get("price_chart_library", "plotille")
        self.price_chart_windows = discovery_config.get("price_chart_windows", ["1m"])
        self.price_chart_lookback_days = discovery_config.get("price_chart_lookback_days", 30)
        self.price_chart_width = discovery_config.get("price_chart_width", 60)
        self.price_chart_height = discovery_config.get("price_chart_height", 12)
        self.price_chart_max_tickers = discovery_config.get("price_chart_max_tickers", 10)
        self.price_chart_show_movement_stats = discovery_config.get(
            "price_chart_show_movement_stats", True
        )

    def _load_logging_config(self) -> None:
        """Load logging configuration."""
        discovery_config = self.config.get("discovery", {})

        self.log_tool_calls = discovery_config.get("log_tool_calls", True)
        self.log_tool_calls_console = discovery_config.get("log_tool_calls_console", False)
        self.tool_log_max_chars = discovery_config.get("tool_log_max_chars", 10_000)
        self.tool_log_exclude = set(discovery_config.get("tool_log_exclude", []))

    def _safe_serialize(self, value: Any) -> str:
        """Safely serialize any value to a string."""
        return serialize_for_log(value)

    def _log_tool_call(
        self,
        tool_logs: List[Dict[str, Any]],
        node: str,
        step_name: str,
        tool_name: str,
        params: Dict[str, Any],
        output: Any,
        context: str = "",
        error: str = "",
    ) -> Dict[str, Any]:
        """
        Log a tool call with metadata for debugging and analysis.

        Args:
            tool_logs: List to append the log entry to
            node: Name of the graph node executing the tool
            step_name: Description of the current step
            tool_name: Name of the tool being executed
            params: Parameters passed to the tool
            output: Output from the tool execution
            context: Additional context for the log entry
            error: Error message if tool execution failed

        Returns:
            The created log entry dictionary
        """
        from datetime import datetime

        output_str = self._safe_serialize(output)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "tool",
            "node": node,
            "step": step_name,
            "tool": tool_name,
            "parameters": params,
            "context": context,
            "output": output_str,
            "output_length": len(output_str),
            "error": error,
        }
        tool_logs.append(log_entry)

        if self.log_tool_calls_console:
            import logging

            output_preview = output_str
            if self.tool_log_max_chars and len(output_preview) > self.tool_log_max_chars:
                output_preview = output_preview[: self.tool_log_max_chars] + "..."
            logging.getLogger(__name__).info(
                "TOOL %s node=%s step=%s params=%s error=%s output=%s",
                tool_name,
                node,
                step_name,
                params,
                bool(error),
                output_preview,
            )

        return log_entry

    def _execute_tool_logged(
        self,
        state: DiscoveryState,
        *,
        node: str,
        step: str,
        tool_name: str,
        context: str = "",
        **params,
    ) -> Any:
        """
        Execute a tool with optional logging.

        Args:
            state: Current discovery state containing tool_logs
            node: Name of the graph node executing the tool
            step: Description of the current step
            tool_name: Name of the tool to execute
            context: Additional context for logging
            **params: Parameters to pass to the tool

        Returns:
            Tool execution result

        Raises:
            Exception: Re-raises any exception from tool execution after logging
        """
        tool_logs = state.get("tool_logs", [])

        if not self.log_tool_calls or tool_name in self.tool_log_exclude:
            return execute_tool(tool_name, **params)

        try:
            result = execute_tool(tool_name, **params)
            self._log_tool_call(
                tool_logs,
                node=node,
                step_name=step,
                tool_name=tool_name,
                params=params,
                output=result,
                context=context,
            )
            state["tool_logs"] = tool_logs
            return result
        except Exception as e:
            self._log_tool_call(
                tool_logs,
                node=node,
                step_name=step,
                tool_name=tool_name,
                params=params,
                output="",
                context=context,
                error=str(e),
            )
            state["tool_logs"] = tool_logs
            raise

    def _create_graph(self) -> StateGraph:
        """
        Create the discovery workflow graph.

        The graph follows this flow:
        scanner -> filter -> ranker -> END

        Returns:
            Compiled workflow graph
        """
        workflow = StateGraph(DiscoveryState)

        workflow.add_node(self.NODE_SCANNER, self.scanner_node)
        workflow.add_node(self.NODE_FILTER, self.filter_node)
        workflow.add_node(self.NODE_RANKER, self.preliminary_ranker_node)

        workflow.set_entry_point(self.NODE_SCANNER)
        workflow.add_edge(self.NODE_SCANNER, self.NODE_FILTER)
        workflow.add_edge(self.NODE_FILTER, self.NODE_RANKER)
        workflow.add_edge(self.NODE_RANKER, END)

        return workflow.compile()

    def semantic_scanner_node(self, state: DiscoveryState) -> Dict[str, Any]:
        """
        Scan market using semantic news-ticker matching.

        Uses news semantic scanner to find tickers mentioned in or
        semantically related to recent market-moving news.

        Args:
            state: Current discovery state

        Returns:
            Updated state with semantic candidates
        """
        print("🔍 Scanning market with semantic discovery...")

        # Update performance tracking for historical recommendations (runs before discovery)
        try:
            self.analytics.update_performance_tracking()
        except Exception as e:
            print(f"   Warning: Performance tracking update failed: {e}")
            print("   Continuing with discovery...")

        tool_logs = state.setdefault("tool_logs", [])

        def log_callback(entry: Dict[str, Any]) -> None:
            tool_logs.append(entry)
            state["tool_logs"] = tool_logs

        try:
            from tradingagents.dataflows.semantic_discovery import SemanticDiscovery

            # Build config for semantic discovery
            semantic_config = {
                "project_dir": self.config.get("project_dir", "."),
                "use_openai_embeddings": True,
                "news_sources": self.semantic_news_sources,
                "max_news_items": 20,
                "news_lookback_hours": self.semantic_news_lookback_hours,
                "min_news_importance": self.semantic_min_news_importance,
                "min_similarity_threshold": self.semantic_min_similarity,
                "max_tickers_per_news": self.semantic_max_tickers_per_news,
                "max_total_candidates": self.max_candidates_to_analyze,
                "log_callback": log_callback,
            }

            # Run semantic discovery
            discovery = SemanticDiscovery(semantic_config)
            ranked_candidates = discovery.discover()

            # Also get directly mentioned tickers from news (highest signal)
            directly_mentioned = discovery.get_directly_mentioned_tickers()

            # Convert to candidate format
            candidates = []

            # Add directly mentioned tickers first (highest priority)
            for ticker_info in directly_mentioned:
                candidates.append(
                    {
                        "ticker": ticker_info["ticker"],
                        "source": self.SOURCE_NEWS_MENTION,
                        "context": f"Directly mentioned in news: {ticker_info['news_title']}",
                        "priority": self.PRIORITY_CRITICAL,  # Direct mention = highest priority
                        "news_sentiment": ticker_info.get("sentiment", "neutral"),
                        "news_importance": ticker_info.get("importance", 5),
                        "news_context": [ticker_info],
                    }
                )

            # Add semantically matched tickers
            for rank_info in ranked_candidates:
                ticker = rank_info["ticker"]
                news_matches = rank_info["news_matches"]

                # Combine all news titles for richer context
                all_news_titles = "; ".join([n["news_title"] for n in news_matches[:3]])

                candidates.append(
                    {
                        "ticker": ticker,
                        "source": self.SOURCE_SEMANTIC,
                        "context": f"News-driven: {all_news_titles}",
                        "priority": self.PRIORITY_HIGH,  # News-driven is always high priority (leading indicator)
                        "semantic_score": rank_info["aggregate_score"],
                        "num_news_matches": rank_info["num_news_matches"],
                        "news_context": news_matches,  # Store full news context for later
                    }
                )

            print(f"   Found {len(candidates)} candidates from semantic discovery.")

            return {
                "tickers": [c["ticker"] for c in candidates],
                "candidate_metadata": candidates,
                "tool_logs": state.get("tool_logs", []),
                "status": "scanned",
            }

        except Exception as e:
            print(f"   Error in semantic discovery: {e}")
            print("   Falling back to traditional scanner...")
            # Directly call traditional scanner to avoid recursion
            return self.traditional_scanner_node(state)

    def _merge_candidates_into_dict(
        self, candidates: List[Dict[str, Any]], target_dict: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Merge candidates into target dictionary with smart deduplication.

        For duplicate tickers, merges sources and contexts intelligently,
        upgrading priority when higher-priority sources are found.

        Args:
            candidates: List of candidate dictionaries to merge
            target_dict: Target dictionary to merge into (ticker -> candidate data)
        """
        for candidate in candidates:
            ticker = candidate["ticker"]

            if ticker not in target_dict:
                self._add_new_candidate(candidate, target_dict)
            else:
                self._merge_with_existing_candidate(candidate, target_dict[ticker])

    def _add_new_candidate(
        self, candidate: Dict[str, Any], target_dict: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Add a new candidate to the target dictionary.

        Args:
            candidate: Candidate dictionary to add
            target_dict: Target dictionary to add to
        """
        ticker = candidate["ticker"]
        target_dict[ticker] = candidate.copy()

        source = candidate.get("source", self.SOURCE_UNKNOWN)
        context = candidate.get("context", "").strip()

        target_dict[ticker]["all_sources"] = [source]
        target_dict[ticker]["all_contexts"] = [context] if context else []

    def _merge_with_existing_candidate(
        self, incoming: Dict[str, Any], existing: Dict[str, Any]
    ) -> None:
        """
        Merge incoming candidate data with existing candidate.

        Args:
            incoming: New candidate data to merge
            existing: Existing candidate data to update
        """
        # Initialize list fields if needed
        existing.setdefault("all_sources", [existing.get("source", self.SOURCE_UNKNOWN)])
        existing.setdefault(
            "all_contexts", [existing.get("context", "")] if existing.get("context") else []
        )

        # Update sources
        incoming_source = incoming.get("source", self.SOURCE_UNKNOWN)
        if incoming_source not in existing["all_sources"]:
            existing["all_sources"].append(incoming_source)

        # Update priority and contexts based on priority ranking
        self._update_priority_and_context(incoming, existing)

    def _update_priority_and_context(
        self, incoming: Dict[str, Any], existing: Dict[str, Any]
    ) -> None:
        """
        Update priority and context based on incoming candidate priority.

        If incoming has higher priority, upgrades existing candidate.
        Otherwise, just appends context.

        Args:
            incoming: New candidate data
            existing: Existing candidate data to update
        """
        incoming_rank = self.PRIORITY_ORDER.get(incoming.get("priority", self.PRIORITY_UNKNOWN), 4)
        existing_rank = self.PRIORITY_ORDER.get(existing.get("priority", self.PRIORITY_UNKNOWN), 4)
        incoming_context = incoming.get("context", "").strip()

        if incoming_rank < existing_rank:
            # Higher priority - upgrade and prepend context
            existing["priority"] = incoming.get("priority")
            existing["source"] = incoming.get("source")
            self._prepend_context(incoming_context, existing)
        else:
            # Same or lower priority - just append context
            self._append_context(incoming_context, existing)

    def _prepend_context(self, new_context: str, candidate: Dict[str, Any]) -> None:
        """
        Prepend context to existing candidate (for higher priority updates).

        Args:
            new_context: New context string to prepend
            candidate: Candidate dictionary to update
        """
        if not new_context:
            return

        candidate["all_contexts"].append(new_context)
        current_ctx = candidate.get("context", "")
        candidate["context"] = f"{new_context}; Also: {current_ctx}" if current_ctx else new_context

    def _append_context(self, new_context: str, candidate: Dict[str, Any]) -> None:
        """
        Append context to existing candidate (for same/lower priority updates).

        Args:
            new_context: New context string to append
            candidate: Candidate dictionary to update
        """
        if not new_context or new_context in candidate["all_contexts"]:
            return

        candidate["all_contexts"].append(new_context)
        current_ctx = candidate.get("context", "")

        if not current_ctx:
            candidate["context"] = new_context
        elif new_context not in current_ctx:
            candidate["context"] = f"{current_ctx}; Also: {new_context}"

    def scanner_node(self, state: DiscoveryState) -> Dict[str, Any]:
        """
        Scan the market for potential candidates using the modular scanner registry.

        Iterates through all scanners in SCANNER_REGISTRY, checks if they're enabled,
        and runs them to collect candidates organized by pipeline.

        Args:
            state: Current discovery state

        Returns:
            Updated state with discovered candidates
        """
        print("Scanning market for opportunities...")

        # Update performance tracking for historical recommendations (runs before discovery)
        try:
            self.analytics.update_performance_tracking()
        except Exception as e:
            print(f"   Warning: Performance tracking update failed: {e}")
            print("   Continuing with discovery...")

        # Initialize tool_logs in state
        state.setdefault("tool_logs", [])

        # Get execution config
        exec_config = self.config.get("discovery", {}).get("scanner_execution", {})
        concurrent = exec_config.get("concurrent", True)
        max_workers = exec_config.get("max_workers", 8)
        timeout_seconds = exec_config.get("timeout_seconds", 30)

        # Get pipeline_config from config
        pipeline_config = self.config.get("discovery", {}).get("pipelines", {})

        # Prepare enabled scanners
        enabled_scanners = []
        for scanner_class in SCANNER_REGISTRY.get_all_scanners():
            pipeline = scanner_class.pipeline

            # Check if scanner's pipeline is enabled
            if not pipeline_config.get(pipeline, {}).get("enabled", True):
                print(f"   Skipping {scanner_class.name} (pipeline '{pipeline}' disabled)")
                continue

            try:
                # Instantiate scanner with config
                scanner = scanner_class(self.config)

                # Check if scanner is enabled
                if not scanner.is_enabled():
                    print(f"   Skipping {scanner_class.name} (scanner disabled)")
                    continue

                enabled_scanners.append((scanner, scanner_class.name, pipeline))

            except Exception as e:
                print(f"   Error instantiating {scanner_class.name}: {e}")
                continue

        # Run scanners concurrently or sequentially based on config
        if concurrent and len(enabled_scanners) > 1:
            pipeline_candidates = self._run_scanners_concurrent(
                enabled_scanners, state, max_workers, timeout_seconds
            )
        else:
            pipeline_candidates = self._run_scanners_sequential(enabled_scanners, state)

        # Merge all candidates from all pipelines using _merge_candidates_into_dict()
        all_candidates_dict: Dict[str, Dict[str, Any]] = {}
        for pipeline, candidates in pipeline_candidates.items():
            self._merge_candidates_into_dict(candidates, all_candidates_dict)

        # Convert merged dict to list
        final_candidates = list(all_candidates_dict.values())
        final_tickers = [c["ticker"] for c in final_candidates]

        print(f"   Found {len(final_candidates)} unique candidates from all scanners.")

        # Return state with tickers, candidate_metadata, tool_logs, status
        return {
            "tickers": final_tickers,
            "candidate_metadata": final_candidates,
            "tool_logs": state.get("tool_logs", []),
            "status": "scanned",
        }

    def _run_scanners_sequential(
        self, enabled_scanners: List[tuple], state: DiscoveryState
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run scanners sequentially (original behavior).

        Args:
            enabled_scanners: List of (scanner, name, pipeline) tuples
            state: Current discovery state

        Returns:
            Dict mapping pipeline -> list of candidates
        """
        pipeline_candidates: Dict[str, List[Dict[str, Any]]] = {}

        for scanner, name, pipeline in enabled_scanners:
            # Initialize pipeline list if needed
            if pipeline not in pipeline_candidates:
                pipeline_candidates[pipeline] = []

            try:
                # Set tool_executor in state for scanner to use
                state["tool_executor"] = self._execute_tool_logged

                # Call scanner.scan_with_validation(state)
                print(f"   Running {name}...")
                candidates = scanner.scan_with_validation(state)

                # Route candidates to appropriate pipeline
                pipeline_candidates[pipeline].extend(candidates)
                print(f"      Found {len(candidates)} candidates")

            except Exception as e:
                print(f"   Error in {name}: {e}")
                continue

        return pipeline_candidates

    def _run_scanners_concurrent(
        self,
        enabled_scanners: List[tuple],
        state: DiscoveryState,
        max_workers: int,
        timeout_seconds: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run scanners concurrently using ThreadPoolExecutor.

        Args:
            enabled_scanners: List of (scanner, name, pipeline) tuples
            state: Current discovery state
            max_workers: Maximum concurrent threads
            timeout_seconds: Timeout per scanner in seconds

        Returns:
            Dict mapping pipeline -> list of candidates
        """
        import logging
        from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

        logger = logging.getLogger(__name__)
        pipeline_candidates: Dict[str, List[Dict[str, Any]]] = {}

        print(
            f"   Running {len(enabled_scanners)} scanners concurrently (max {max_workers} workers)..."
        )

        def run_scanner(scanner_info: tuple) -> tuple:
            """Execute a single scanner with error handling."""
            scanner, name, pipeline = scanner_info
            try:
                # Create a copy of state for thread safety
                scanner_state = state.copy()
                scanner_state["tool_executor"] = self._execute_tool_logged

                # Run scanner with validation
                candidates = scanner.scan_with_validation(scanner_state)

                # Merge tool_logs back into main state (thread-safe append)
                if "tool_logs" in scanner_state:
                    state.setdefault("tool_logs", []).extend(scanner_state["tool_logs"])

                return (name, pipeline, candidates, None)

            except Exception as e:
                logger.error(f"Scanner {name} failed: {e}", exc_info=True)
                return (name, pipeline, [], str(e))

        # Submit all scanner tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_scanner = {
                executor.submit(run_scanner, scanner_info): scanner_info[1]
                for scanner_info in enabled_scanners
            }

            # Collect results as they complete (no global timeout, handle per-scanner)
            completed_count = 0
            for future in as_completed(future_to_scanner):
                scanner_name = future_to_scanner[future]

                try:
                    # Get result with per-scanner timeout
                    name, pipeline, candidates, error = future.result(timeout=timeout_seconds)

                    # Initialize pipeline list if needed
                    if pipeline not in pipeline_candidates:
                        pipeline_candidates[pipeline] = []

                    if error:
                        print(f"   ⚠️  {name}: {error}")
                    else:
                        pipeline_candidates[pipeline].extend(candidates)
                        print(f"   ✓ {name}: {len(candidates)} candidates")

                except TimeoutError:
                    logger.warning(f"Scanner {scanner_name} timed out after {timeout_seconds}s")
                    print(f"   ⏱️  {scanner_name}: timeout after {timeout_seconds}s")

                except Exception as e:
                    logger.error(f"Scanner {scanner_name} failed unexpectedly: {e}", exc_info=True)
                    print(f"   ⚠️  {scanner_name}: unexpected error")

                finally:
                    completed_count += 1

            # Log completion stats
            if completed_count < len(enabled_scanners):
                logger.warning(f"Only {completed_count}/{len(enabled_scanners)} scanners completed")

        return pipeline_candidates

    def hybrid_scanner_node(self, state: DiscoveryState) -> Dict[str, Any]:
        """
        Run both semantic and traditional discovery with smart deduplication.

        Combines news-driven semantic discovery (leading indicators) with
        traditional discovery (social, market movers, earnings). Merges
        results and boosts candidates confirmed by multiple sources.

        Args:
            state: Current discovery state

        Returns:
            Updated state with merged candidates from both approaches
        """
        print("🔍 Hybrid Discovery: Combining news-driven AND traditional signals...")

        # Update performance tracking once (not in each sub-scanner)
        try:
            self.analytics.update_performance_tracking()
        except Exception as e:
            print(f"   Warning: Performance tracking update failed: {e}")
            print("   Continuing with discovery...")

        tool_logs = state.setdefault("tool_logs", [])

        def log_callback(entry: Dict[str, Any]) -> None:
            tool_logs.append(entry)
            state["tool_logs"] = tool_logs

        # We will merge all candidates into this dict
        unique_candidates = {}
        all_tickers = set()

        # ========================================
        # Phase 1: Semantic Discovery (news-driven - leading indicators)
        # ========================================
        print("\n📰 Phase 1: Semantic Discovery (news-driven)...")
        try:
            from tradingagents.dataflows.semantic_discovery import SemanticDiscovery

            # Build config for semantic discovery
            semantic_config = {
                "project_dir": self.config.get("project_dir", "."),
                "use_openai_embeddings": True,
                "news_sources": self.semantic_news_sources,
                "max_news_items": 20,
                "news_lookback_hours": self.semantic_news_lookback_hours,
                "min_news_importance": self.semantic_min_news_importance,
                "min_similarity_threshold": self.semantic_min_similarity,
                "max_tickers_per_news": self.semantic_max_tickers_per_news,
                "max_total_candidates": self.max_candidates_to_analyze,
                "log_callback": log_callback,
            }

            # Run semantic discovery
            discovery = SemanticDiscovery(semantic_config)
            ranked_candidates = discovery.discover()

            # Also get directly mentioned tickers from news (highest signal)
            directly_mentioned = discovery.get_directly_mentioned_tickers()

            # Prepare semantic candidates list
            semantic_candidates = []

            # Add directly mentioned tickers first (highest priority)
            for ticker_info in directly_mentioned:
                semantic_candidates.append(
                    {
                        "ticker": ticker_info["ticker"],
                        "source": self.SOURCE_NEWS_MENTION,
                        "context": f"Directly mentioned in news: {ticker_info['news_title']}",
                        "priority": self.PRIORITY_CRITICAL,  # Direct mention = highest priority
                        "news_sentiment": ticker_info.get("sentiment", "neutral"),
                        "news_importance": ticker_info.get("importance", 5),
                        "news_context": [ticker_info],
                    }
                )
                all_tickers.add(ticker_info["ticker"])

            # Add semantically matched tickers
            for rank_info in ranked_candidates:
                ticker = rank_info["ticker"]
                news_matches = rank_info["news_matches"]

                # Combine all news titles for richer context
                all_news_titles = "; ".join([n["news_title"] for n in news_matches[:3]])

                semantic_candidates.append(
                    {
                        "ticker": ticker,
                        "source": self.SOURCE_SEMANTIC,
                        "context": f"News-driven: {all_news_titles}",
                        "priority": self.PRIORITY_HIGH,  # News-driven is always high priority (leading indicator)
                        "semantic_score": rank_info["aggregate_score"],
                        "num_news_matches": rank_info["num_news_matches"],
                        "news_context": news_matches,
                    }
                )
                all_tickers.add(ticker)

            print(f"   Found {len(semantic_candidates)} candidates from semantic discovery")

            # Merge semantic candidates into unique dict
            self._merge_candidates_into_dict(semantic_candidates, unique_candidates)

        except Exception as e:
            print(f"   Semantic discovery failed: {e}")
            print("   Continuing with traditional discovery...")

        # ========================================
        # Phase 2: Traditional Discovery (social, market movers, etc.)
        # ========================================
        print("\n📊 Phase 2: Traditional Discovery (Reddit, market movers, earnings, etc.)...")
        traditional_candidates = self._run_traditional_scanners(state)
        print(f"   Found {len(traditional_candidates)} candidates from traditional discovery")

        # Merge traditional candidates into unique dict
        self._merge_candidates_into_dict(traditional_candidates, unique_candidates)

        # ========================================
        # Phase 3: Post-Merge Processing
        # ========================================
        print("\n🔄 Phase 3: Finalizing candidates...")

        final_candidates = list(unique_candidates.values())

        # Check for multi-source confirmation
        semantic_sources = {self.SOURCE_SEMANTIC, self.SOURCE_NEWS_MENTION}

        for c in final_candidates:
            sources = c.get("all_sources", [])
            has_semantic = any(s in semantic_sources for s in sources)
            has_traditional = any(
                s not in semantic_sources and s != self.SOURCE_UNKNOWN for s in sources
            )

            if has_semantic and has_traditional:
                # Found by BOTH semantic and traditional - boost confidence
                c["multi_source_confirmed"] = True
                if c.get("priority") == self.PRIORITY_HIGH:
                    c["priority"] = self.PRIORITY_CRITICAL  # Upgrade to critical

        # Sort by priority
        final_candidates.sort(
            key=lambda x: self.PRIORITY_ORDER.get(x.get("priority", self.PRIORITY_UNKNOWN), 4)
        )

        # Update all_tickers set
        all_tickers = {c["ticker"] for c in final_candidates}

        # Count by priority for reporting
        critical_count = sum(
            1 for c in final_candidates if c.get("priority") == self.PRIORITY_CRITICAL
        )
        high_count = sum(1 for c in final_candidates if c.get("priority") == self.PRIORITY_HIGH)
        medium_count = sum(1 for c in final_candidates if c.get("priority") == self.PRIORITY_MEDIUM)
        low_count = sum(1 for c in final_candidates if c.get("priority") == self.PRIORITY_LOW)
        multi_confirmed = sum(1 for c in final_candidates if c.get("multi_source_confirmed"))

        print(f"\n✅ Hybrid discovery complete: {len(final_candidates)} total candidates")
        print(
            f"   Priority: {critical_count} critical, {high_count} high, {medium_count} medium, {low_count} low"
        )
        if multi_confirmed:
            print(
                f"   🎯 {multi_confirmed} candidates confirmed by BOTH semantic AND traditional sources"
            )

        return {
            "tickers": list(all_tickers),
            "candidate_metadata": final_candidates,
            "tool_logs": state.get("tool_logs", []),
            "status": "scanned",
        }

    def _run_traditional_scanners(self, state: DiscoveryState) -> List[Dict[str, Any]]:
        """
        Run all traditional scanner sources and return candidates.

        Traditional sources include:
        - Reddit trending
        - Market movers
        - Earnings calendar
        - IPO calendar
        - Short interest
        - Unusual volume
        - Analyst rating changes
        - Insider buying

        Args:
            state: Current discovery state

        Returns:
            List of candidates (without deduplication)
        """
        from tradingagents.dataflows.discovery.scanners import TraditionalScanner

        scanner = TraditionalScanner(
            config=self.config, llm=self.quick_thinking_llm, tool_executor=self._execute_tool_logged
        )
        return scanner.scan(state)

    def traditional_scanner_node(self, state: DiscoveryState) -> Dict[str, Any]:
        """
        Traditional market scanning: Reddit, market movers, earnings, etc.

        Args:
            state: Current discovery state

        Returns:
            Updated state with traditional candidates
        """
        print("🔍 Scanning market for opportunities...")

        # Update performance tracking for historical recommendations (runs before discovery)
        try:
            self.analytics.update_performance_tracking()
        except Exception as e:
            print(f"   Warning: Performance tracking update failed: {e}")
            print("   Continuing with discovery...")

        state.setdefault("tool_logs", [])

        # Run all traditional scanners
        candidates = self._run_traditional_scanners(state)

        # Deduplicate candidates
        unique_candidates = {}
        self._merge_candidates_into_dict(candidates, unique_candidates)

        final_candidates = list(unique_candidates.values())
        print(f"   Found {len(final_candidates)} unique candidates.")

        return {
            "tickers": [c["ticker"] for c in final_candidates],
            "candidate_metadata": final_candidates,
            "tool_logs": state.get("tool_logs", []),
            "status": "scanned",
        }

    def filter_node(self, state: DiscoveryState) -> Dict[str, Any]:
        """
        Filter candidates and enrich with additional data.

        Filters candidates based on:
        - Ticker validity
        - Liquidity (volume)
        - Same-day price movement
        - Data availability

        Enriches with:
        - Current price
        - Fundamentals
        - Business description
        - Technical indicators
        - News
        - Insider transactions
        - Analyst recommendations
        - Options activity

        Args:
            state: Current discovery state with candidates

        Returns:
            Updated state with filtered and enriched candidates
        """
        from tradingagents.dataflows.discovery.filter import CandidateFilter

        cand_filter = CandidateFilter(self.config, self._execute_tool_logged)
        return cand_filter.filter(state)

    def preliminary_ranker_node(self, state: DiscoveryState) -> Dict[str, Any]:
        """
        Rank all filtered candidates and select top opportunities.

        Uses LLM to analyze all enriched candidate data and rank
        by investment potential based on:
        - Strategy match
        - Fundamental strength
        - Technical setup
        - Catalyst timing
        - Options flow
        - Historical performance patterns

        Args:
            state: Current discovery state with filtered candidates

        Returns:
            Final state with ranked opportunities and final_ranking JSON
        """
        from tradingagents.dataflows.discovery.ranker import CandidateRanker

        ranker = CandidateRanker(self.config, self.deep_thinking_llm, self.analytics)
        return ranker.rank(state)

    def run(self, trade_date: str = None):
        """Execute the discovery graph workflow.

        Args:
            trade_date: Trade date in YYYY-MM-DD format (defaults to today if not provided)
        """
        from tradingagents.dataflows.discovery.utils import resolve_trade_date_str

        trade_date = resolve_trade_date_str({"trade_date": trade_date})

        print(f"\n{'='*60}")
        print(f"Discovery Analysis - {trade_date}")
        print(f"{'='*60}")

        initial_state = {
            "trade_date": trade_date,
            "tickers": [],
            "filtered_tickers": [],
            "final_ranking": "",
            "status": "initialized",
            "tool_logs": [],
        }

        final_state = self.graph.invoke(initial_state)

        # Save results and recommendations
        self.analytics.save_discovery_results(final_state, trade_date, self.config)

        # Extract and save rankings if available
        rankings = final_state.get("final_ranking", [])
        if isinstance(rankings, str):
            try:
                import json

                rankings = json.loads(rankings)
            except Exception:
                rankings = []
        if rankings:
            if isinstance(rankings, dict) and "rankings" in rankings:
                rankings_list = rankings["rankings"]
            elif isinstance(rankings, list):
                rankings_list = rankings
            else:
                rankings_list = []

            if rankings_list:
                self.analytics.save_recommendations(
                    rankings_list, trade_date, self.config.get("llm_provider", "unknown")
                )

        return final_state

    def build_price_chart_bundle(self, rankings: Any) -> Dict[str, Dict[str, Any]]:
        """Build per-ticker chart + movement stats for top recommendations."""
        if not self.console_price_charts:
            return {}

        rankings_list = self._normalize_rankings(rankings)
        tickers: List[str] = []
        for item in rankings_list:
            ticker = (item.get("ticker") or "").upper()
            if ticker and ticker not in tickers:
                tickers.append(ticker)

        if not tickers:
            return {}

        tickers = tickers[: self.price_chart_max_tickers]
        chart_windows = self._get_chart_windows()
        renderer = self._get_chart_renderer()
        if renderer is None:
            return {}

        bundle: Dict[str, Dict[str, Any]] = {}
        for ticker in tickers:
            series = self._fetch_price_series(ticker)
            if not series:
                bundle[ticker] = {
                    "chart": f"{ticker}: no price history available",
                    "charts": {},
                    "movement": {},
                }
                continue

            per_window_charts: Dict[str, str] = {}
            for window in chart_windows:
                window_closes = self._get_window_closes(ticker, series, window)
                if len(window_closes) < 2:
                    continue

                change_pct = None
                if window_closes[0]:
                    change_pct = (window_closes[-1] / window_closes[0] - 1) * 100.0

                label = window.upper()
                title = f"{ticker} ({label})"
                if change_pct is not None:
                    title = f"{ticker} ({label}, {change_pct:+.1f}%)"

                chart_text = renderer(window_closes, title)
                if chart_text:
                    per_window_charts[window] = chart_text

            primary_chart = ""
            if per_window_charts:
                first_key = chart_windows[0]
                primary_chart = per_window_charts.get(
                    first_key, next(iter(per_window_charts.values()))
                )

            bundle[ticker] = {
                "chart": primary_chart,
                "charts": per_window_charts,
                "movement": self._compute_movement_stats(series),
            }
        return bundle

    def build_price_chart_map(self, rankings: Any) -> Dict[str, str]:
        """Build mini price charts keyed by ticker."""
        bundle = self.build_price_chart_bundle(rankings)
        return {ticker: item.get("chart", "") for ticker, item in bundle.items()}

    def build_price_chart_strings(self, rankings: Any) -> List[str]:
        """Build mini price charts for top recommendations (returns ANSI strings)."""
        charts = self.build_price_chart_map(rankings)
        return list(charts.values()) if charts else []

    def _print_price_charts(self, rankings_list: List[Dict[str, Any]]) -> None:
        """Render mini price charts for top recommendations in the console."""
        charts = self.build_price_chart_strings(rankings_list)
        if not charts:
            return

        print(f"\n📈 Price Charts (last {self.price_chart_lookback_days} days)")
        for chart in charts:
            print(chart)

    def _fetch_price_series(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch recent daily close prices with dates for charting and movement stats."""
        try:
            import pandas as pd
            import yfinance as yf

            from tradingagents.dataflows.y_finance import suppress_yfinance_warnings

            history_days = max(self.price_chart_lookback_days + 10, 390)
            with suppress_yfinance_warnings():
                data = yf.download(
                    ticker,
                    period=f"{history_days}d",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )

            if data is None or data.empty:
                return []

            series = None
            if isinstance(data.columns, pd.MultiIndex):
                if "Close" in data.columns.get_level_values(0):
                    close_data = data["Close"]
                    series = (
                        close_data.iloc[:, 0]
                        if isinstance(close_data, pd.DataFrame)
                        else close_data
                    )
            elif "Close" in data.columns:
                series = data["Close"]

            if series is None:
                series = data.iloc[:, 0]

            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]

            series = series.dropna()
            if series.empty:
                return []

            points: List[Dict[str, Any]] = []
            for index, close in series.items():
                dt = getattr(index, "to_pydatetime", lambda: index)()
                points.append({"date": dt, "close": float(close)})
            return points
        except Exception as exc:
            print(f"   {ticker}: error fetching prices: {exc}")
            return []

    def _get_chart_renderer(self) -> Optional[Callable[[List[float], str], str]]:
        """Return selected chart renderer, with fallback to plotext."""
        preferred = str(self.price_chart_library or "plotext").lower().strip()

        if preferred == "plotille":
            try:
                import plotille

                return lambda closes, title: self._render_plotille_chart(plotille, closes, title)
            except Exception as exc:
                print(f"   ⚠️  plotille unavailable, falling back to plotext: {exc}")

        try:
            import plotext as plt

            return lambda closes, title: self._render_plotext_chart(plt, closes, title)
        except Exception as exc:
            print(f"   ⚠️  plotext not available, skipping charts: {exc}")
            return None

    def _render_plotille_chart(self, plotille: Any, closes: List[float], title: str) -> str:
        """Build a plotille chart and return as ANSI string."""
        if not closes:
            return ""

        fig = plotille.Figure()
        fig.width = self.price_chart_width
        fig.height = self.price_chart_height
        fig.color_mode = "byte"
        fig.set_x_limits(min_=0, max_=max(1, len(closes) - 1))

        min_close = min(closes)
        max_close = max(closes)
        if min_close == max_close:
            padding = max(0.01, min_close * 0.01)
            min_close -= padding
            max_close += padding
        fig.set_y_limits(min_=min_close, max_=max_close)
        fig.plot(range(len(closes)), closes, lc=45)

        return f"{title}\n{fig.show(legend=False)}"

    def _render_plotext_chart(self, plt: Any, closes: List[float], title: str) -> str:
        """Build a single plotext line chart and return as ANSI string."""
        self._reset_plotext(plt)

        if hasattr(plt, "plotsize"):
            plt.plotsize(self.price_chart_width, self.price_chart_height)

        if hasattr(plt, "theme"):
            try:
                plt.theme("pro")
            except Exception:
                pass

        if hasattr(plt, "title"):
            plt.title(title)

        if hasattr(plt, "xlabel"):
            plt.xlabel("")
        if hasattr(plt, "ylabel"):
            plt.ylabel("")

        plt.plot(closes)

        if hasattr(plt, "build"):
            chart = plt.build()
            if chart:
                return chart

        plt.show()
        return ""

    def _compute_movement_stats(self, series: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Compute 1D, 7D, 6M, and 1Y percent movement from latest close."""
        if not series:
            return {}

        from datetime import timedelta

        latest = series[-1]
        latest_date = latest["date"]
        latest_close = latest["close"]

        if not latest_close:
            return {}

        windows = {
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "1m": timedelta(days=30),
            "6m": timedelta(days=182),
            "1y": timedelta(days=365),
        }

        stats: Dict[str, Optional[float]] = {}
        for label, delta in windows.items():
            target_date = latest_date - delta
            baseline = None
            for point in series:
                if point["date"] <= target_date:
                    baseline = point["close"]
                else:
                    break

            if baseline and baseline != 0:
                stats[label] = (latest_close / baseline - 1.0) * 100.0
            else:
                stats[label] = None
        return stats

    def _get_chart_windows(self) -> List[str]:
        """Normalize configured chart windows."""
        allowed = {"1d", "7d", "1m", "6m", "1y"}
        configured = self.price_chart_windows
        if isinstance(configured, str):
            configured = [part.strip().lower() for part in configured.split(",")]
        elif not isinstance(configured, list):
            configured = ["1m"]

        windows = []
        for value in configured:
            key = str(value).strip().lower()
            if key in allowed and key not in windows:
                windows.append(key)
        return windows or ["1m"]

    def _get_window_closes(
        self, ticker: str, series: List[Dict[str, Any]], window: str
    ) -> List[float]:
        """Return closes for a given chart window."""
        if not series:
            return []

        from datetime import timedelta

        if window == "1d":
            intraday = self._fetch_intraday_closes(ticker)
            if len(intraday) >= 2:
                return intraday
            # fallback to last 2 daily points if intraday unavailable
            return [point["close"] for point in series[-2:]]

        window_days = {
            "7d": 7,
            "1m": 30,
            "6m": 182,
            "1y": 365,
        }.get(window, self.price_chart_lookback_days)

        latest_date = series[-1]["date"]
        cutoff = latest_date - timedelta(days=window_days)
        closes = [point["close"] for point in series if point["date"] >= cutoff]
        return closes

    def _fetch_intraday_closes(self, ticker: str) -> List[float]:
        """Fetch intraday close prices for 1-day chart window."""
        try:
            import pandas as pd
            import yfinance as yf

            from tradingagents.dataflows.y_finance import suppress_yfinance_warnings

            with suppress_yfinance_warnings():
                data = yf.download(
                    ticker,
                    period="1d",
                    interval="15m",
                    auto_adjust=True,
                    progress=False,
                )

            if data is None or data.empty:
                return []

            series = None
            if isinstance(data.columns, pd.MultiIndex):
                if "Close" in data.columns.get_level_values(0):
                    close_data = data["Close"]
                    series = (
                        close_data.iloc[:, 0]
                        if isinstance(close_data, pd.DataFrame)
                        else close_data
                    )
            elif "Close" in data.columns:
                series = data["Close"]

            if series is None:
                series = data.iloc[:, 0]

            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]

            return [float(value) for value in series.dropna().to_list()]
        except Exception:
            return []

    @staticmethod
    def _normalize_rankings(rankings: Any) -> List[Dict[str, Any]]:
        """Normalize ranking payload into a list of ranking dicts."""
        rankings_list: List[Dict[str, Any]] = []
        if isinstance(rankings, str):
            try:
                import json

                rankings = json.loads(rankings)
            except Exception:
                rankings = []
        if isinstance(rankings, dict):
            rankings_list = rankings.get("rankings", [])
        elif isinstance(rankings, list):
            rankings_list = rankings
        return rankings_list

    @staticmethod
    def _reset_plotext(plt: Any) -> None:
        """Clear plotext state between charts."""
        for method in ("clf", "clear_figure", "clear_data"):
            func = getattr(plt, method, None)
            if callable(func):
                func()
                return
