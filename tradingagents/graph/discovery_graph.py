from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List

from langgraph.graph import END, StateGraph

from tradingagents.agents.utils.agent_states import DiscoveryState
from tradingagents.dataflows.discovery.discovery_config import DiscoveryConfig
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY
from tradingagents.dataflows.discovery.utils import PRIORITY_ORDER, Priority, serialize_for_log
from tradingagents.tools.executor import execute_tool
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from tradingagents.graph.price_charts import PriceChartBuilder


class DiscoveryGraph:
    """
    Discovery Graph for finding investment opportunities.

    Orchestrates the discovery workflow: scanning -> filtering -> ranking.
    Uses the modular scanner registry to discover candidates.
    """

    # Node names
    NODE_SCANNER = "scanner"
    NODE_FILTER = "filter"
    NODE_RANKER = "ranker"

    # Source types
    SOURCE_UNKNOWN = "unknown"

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
        self._tool_logs_lock = Lock()  # Thread-safe state mutation lock

        # Load scanner modules to trigger registration
        from tradingagents.dataflows.discovery import scanners

        _ = scanners  # Ensure scanners module is loaded

        # Initialize LLMs
        from tradingagents.utils.llm_factory import create_llms

        try:
            self.deep_thinking_llm, self.quick_thinking_llm = create_llms(self.config)
        except Exception as e:
            logger.error(f"Failed to initialize LLMs: {e}")
            raise ValueError(
                f"LLM initialization failed. Check your config's llm_provider setting. Error: {e}"
            ) from e

        # Load typed discovery configuration
        self.dc = DiscoveryConfig.from_config(self.config)

        # Alias frequently-used config for downstream compatibility
        self.log_tool_calls = self.dc.logging.log_tool_calls
        self.log_tool_calls_console = self.dc.logging.log_tool_calls_console
        self.tool_log_max_chars = self.dc.logging.tool_log_max_chars
        self.tool_log_exclude = set(self.dc.logging.tool_log_exclude)

        # Store run directory for saving results
        self.run_dir = self.config.get("discovery_run_dir", None)

        # Initialize Analytics
        from tradingagents.dataflows.discovery.analytics import DiscoveryAnalytics

        self.analytics = DiscoveryAnalytics(data_dir="data")

        self.graph = self._create_graph()

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

        output_str = serialize_for_log(output)

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
            output_preview = output_str
            if self.tool_log_max_chars and len(output_preview) > self.tool_log_max_chars:
                output_preview = output_preview[: self.tool_log_max_chars] + "..."
            logger.info(
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

    def _update_performance_tracking(self) -> None:
        """Update performance tracking for historical recommendations (runs before discovery)."""
        try:
            self.analytics.update_performance_tracking()
        except Exception as e:
            logger.warning(f"Performance tracking update failed: {e}")
            logger.warning("Continuing with discovery...")

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
                # First time seeing this ticker - initialize tracking fields
                entry = candidate.copy()
                source = candidate.get("source", self.SOURCE_UNKNOWN)
                context = candidate.get("context", "").strip()
                entry["all_sources"] = [source]
                entry["all_contexts"] = [context] if context else []
                target_dict[ticker] = entry
            else:
                # Duplicate ticker - merge sources, contexts, and priority
                existing = target_dict[ticker]
                existing.setdefault("all_sources", [existing.get("source", self.SOURCE_UNKNOWN)])
                existing.setdefault(
                    "all_contexts",
                    [existing.get("context", "")] if existing.get("context") else [],
                )

                incoming_source = candidate.get("source", self.SOURCE_UNKNOWN)
                if incoming_source not in existing["all_sources"]:
                    existing["all_sources"].append(incoming_source)

                incoming_context = candidate.get("context", "").strip()
                incoming_rank = PRIORITY_ORDER.get(
                    candidate.get("priority", Priority.UNKNOWN.value), 4
                )
                existing_rank = PRIORITY_ORDER.get(
                    existing.get("priority", Priority.UNKNOWN.value), 4
                )

                if incoming_rank < existing_rank:
                    # Higher priority incoming - upgrade and prepend context
                    existing["priority"] = candidate.get("priority")
                    existing["source"] = candidate.get("source")
                    self._add_context(incoming_context, existing, prepend=True)
                else:
                    self._add_context(incoming_context, existing, prepend=False)

    def _add_context(self, new_context: str, candidate: Dict[str, Any], *, prepend: bool) -> None:
        """
        Add context string to a candidate's context fields.

        When prepend is True, the new context leads the combined string
        (used when a higher-priority source is being merged in).

        Args:
            new_context: New context string to add
            candidate: Candidate dictionary to update
            prepend: If True, new context leads the combined string
        """
        if not new_context or new_context in candidate["all_contexts"]:
            return

        candidate["all_contexts"].append(new_context)
        current_ctx = candidate.get("context", "")

        if not current_ctx:
            candidate["context"] = new_context
        elif new_context not in current_ctx:
            if prepend:
                candidate["context"] = f"{new_context}; Also: {current_ctx}"
            else:
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
        logger.info("Scanning market for opportunities...")

        self._update_performance_tracking()
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
                logger.info(f"Skipping {scanner_class.name} (pipeline '{pipeline}' disabled)")
                continue

            try:
                # Instantiate scanner with config
                scanner = scanner_class(self.config)

                # Check if scanner is enabled
                if not scanner.is_enabled():
                    logger.info(f"Skipping {scanner_class.name} (scanner disabled)")
                    continue

                enabled_scanners.append((scanner, scanner_class.name, pipeline))

            except Exception as e:
                logger.error(f"Error instantiating {scanner_class.name}: {e}")
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

        logger.info(f"Found {len(final_candidates)} unique candidates from all scanners.")

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
                logger.info(f"Running {name}...")
                candidates = scanner.scan_with_validation(state)

                # Route candidates to appropriate pipeline
                pipeline_candidates[pipeline].extend(candidates)
                logger.info(f"Found {len(candidates)} candidates")

            except Exception as e:
                logger.error(f"Error in {name}: {e}")
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
        from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

        pipeline_candidates: Dict[str, List[Dict[str, Any]]] = {}

        # Global wall-clock limit: all scanners must finish within this budget.
        # Using timeout_seconds as per-scanner budget × number of scanners gives a
        # reasonable upper bound, capped at 5 minutes so a single slow scanner can
        # never block the whole run indefinitely.
        global_timeout = min(timeout_seconds * len(enabled_scanners), 300)

        logger.info(
            f"Running {len(enabled_scanners)} scanners concurrently "
            f"(max {max_workers} workers, global timeout {global_timeout}s)..."
        )

        def run_scanner(scanner_info: tuple) -> tuple:
            """Execute a single scanner with error handling."""
            scanner, name, pipeline = scanner_info
            try:
                # Create a copy of state for thread safety
                scanner_state = state.copy()
                scanner_state["tool_logs"] = []  # Fresh log list
                scanner_state["tool_executor"] = self._execute_tool_logged

                # Run scanner with validation
                candidates = scanner.scan_with_validation(scanner_state)

                # Return logs to be merged later (not in-place)
                scanner_logs = scanner_state.get("tool_logs", [])
                return (name, pipeline, candidates, None, scanner_logs)

            except Exception as e:
                logger.error(f"Scanner {name} failed: {e}", exc_info=True)
                return (name, pipeline, [], str(e), [])

        # Submit all scanner tasks.
        # NOTE: Do NOT use `with ThreadPoolExecutor() as executor` here — that
        # form calls shutdown(wait=True) on exit, which blocks until every thread
        # finishes even after as_completed() has already timed out.  We call
        # shutdown(wait=False) explicitly so stuck threads are abandoned.
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            future_to_scanner = {
                executor.submit(run_scanner, scanner_info): scanner_info[1]
                for scanner_info in enabled_scanners
            }

            # Collect results as they complete.
            # global_timeout is the wall-clock budget for ALL scanners together.
            # If any thread blocks indefinitely (e.g. a hung yfinance download),
            # as_completed() raises TimeoutError so we continue immediately.
            completed_count = 0
            try:
                for future in as_completed(future_to_scanner, timeout=global_timeout):
                    scanner_name = future_to_scanner[future]

                    try:
                        name, pipeline, candidates, error, scanner_logs = future.result()

                        # Initialize pipeline list if needed
                        if pipeline not in pipeline_candidates:
                            pipeline_candidates[pipeline] = []

                        if error:
                            logger.warning(f"⚠️ {name}: {error}")
                        else:
                            pipeline_candidates[pipeline].extend(candidates)
                            logger.info(f"✓ {name}: {len(candidates)} candidates")

                        # Thread-safe log merging
                        if scanner_logs:
                            with self._tool_logs_lock:
                                state.setdefault("tool_logs", []).extend(scanner_logs)

                    except Exception as e:
                        logger.error(f"⚠️ {scanner_name}: unexpected error - {e}", exc_info=True)

                    finally:
                        completed_count += 1

            except TimeoutError:
                # Identify which scanners did not finish in time
                stuck = [name for fut, name in future_to_scanner.items() if not fut.done()]
                logger.warning(
                    f"⏱️ Global scanner timeout ({global_timeout}s) reached. "
                    f"Timed-out scanners: {stuck}. Continuing with {completed_count} completed."
                )

            # Log completion stats
            if completed_count < len(enabled_scanners):
                logger.warning(f"Only {completed_count}/{len(enabled_scanners)} scanners completed")

        finally:
            # wait=False: don't block on threads that are still running (e.g. a
            # hung ml_signal download).  The daemon threads will be cleaned up
            # when the process exits.
            executor.shutdown(wait=False)

        return pipeline_candidates

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

        logger.info(f"\n{'='*60}")
        logger.info(f"Discovery Analysis - {trade_date}")
        logger.info(f"{'='*60}")

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
        rankings_list = self._normalize_rankings(final_state.get("final_ranking", []))
        if rankings_list:
            self.analytics.save_recommendations(
                rankings_list, trade_date, self.config.get("llm_provider", "unknown")
            )

        return final_state

    # ------------------------------------------------------------------
    # Price chart delegation (implementation in price_charts.py)
    # ------------------------------------------------------------------

    def _get_chart_builder(self) -> PriceChartBuilder:
        """Lazily create and cache the PriceChartBuilder instance."""
        if not hasattr(self, "_chart_builder"):
            from tradingagents.graph.price_charts import PriceChartBuilder

            c = self.dc.charts
            self._chart_builder = PriceChartBuilder(
                enabled=c.enabled,
                library=c.library,
                windows=c.windows,
                lookback_days=c.lookback_days,
                width=c.width,
                height=c.height,
                max_tickers=c.max_tickers,
                show_movement_stats=c.show_movement_stats,
            )
        return self._chart_builder

    def build_price_chart_bundle(self, rankings: Any) -> Dict[str, Dict[str, Any]]:
        """Build per-ticker chart + movement stats for top recommendations."""
        return self._get_chart_builder().build_bundle(self._normalize_rankings(rankings))

    def build_price_chart_map(self, rankings: Any) -> Dict[str, str]:
        """Build mini price charts keyed by ticker."""
        return self._get_chart_builder().build_map(self._normalize_rankings(rankings))

    def build_price_chart_strings(self, rankings: Any) -> List[str]:
        """Build mini price charts for top recommendations (returns ANSI strings)."""
        return self._get_chart_builder().build_strings(self._normalize_rankings(rankings))

    def _print_price_charts(self, rankings_list: List[Dict[str, Any]]) -> None:
        """Render mini price charts for top recommendations in the console."""
        self._get_chart_builder().print_charts(rankings_list)

    @staticmethod
    def _normalize_rankings(rankings: Any) -> List[Dict[str, Any]]:
        """Normalize ranking payload into a list of ranking dicts."""
        if isinstance(rankings, str):
            try:
                import json

                parsed = json.loads(rankings)
                # Validate parsed result is expected type
                if isinstance(parsed, dict):
                    return parsed.get("rankings", [])
                elif isinstance(parsed, list):
                    return parsed
                else:
                    logger.warning(f"Unexpected JSON type after parsing: {type(parsed)}")
                    return []
            except Exception as e:
                logger.warning(f"Failed to parse rankings JSON: {e}")
                return []
        if isinstance(rankings, dict):
            return rankings.get("rankings", [])
        if isinstance(rankings, list):
            return rankings
        logger.warning(f"Unexpected rankings type: {type(rankings)}")
        return []
