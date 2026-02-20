from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class BaseScanner(ABC):
    """Base class for all discovery scanners."""

    name: str = None
    pipeline: str = None
    strategy: str = None

    def __init__(self, config: Dict[str, Any]):
        if self.name is None:
            raise ValueError(f"{self.__class__.__name__} must define 'name'")
        if self.pipeline is None:
            raise ValueError(f"{self.__class__.__name__} must define 'pipeline'")
        if self.strategy is None:
            raise ValueError(f"{self.__class__.__name__} must define 'strategy'")

        self.config = config
        self.scanner_config = config.get("discovery", {}).get("scanners", {}).get(self.name, {})
        self.enabled = self.scanner_config.get("enabled", True)
        self.limit = self.scanner_config.get("limit", 10)

    @abstractmethod
    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return list of candidates with: ticker, source, context, priority"""
        pass

    def scan_with_validation(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan and validate output format.

        Wraps scan() to validate all candidates have required keys and valid formats.
        Invalid candidates are filtered out and logged.

        Args:
            state: Discovery state dictionary

        Returns:
            List of validated candidates
        """
        try:
            candidates = self.scan(state)

            if not isinstance(candidates, list):
                logger.error(f"{self.name}: scan() returned {type(candidates)}, expected list")
                return []

            # Validate each candidate
            from tradingagents.dataflows.discovery.common_utils import validate_candidate_structure

            valid_candidates = []
            for i, candidate in enumerate(candidates):
                if validate_candidate_structure(candidate):
                    valid_candidates.append(candidate)
                else:
                    logger.warning(
                        f"{self.name}: Invalid candidate #{i}: {candidate}",
                        extra={"scanner": self.name, "pipeline": self.pipeline},
                    )

            if len(valid_candidates) < len(candidates):
                filtered_count = len(candidates) - len(valid_candidates)
                logger.info(
                    f"{self.name}: Filtered {filtered_count}/{len(candidates)} invalid candidates"
                )

            return valid_candidates

        except Exception as e:
            logger.error(
                f"{self.name}: Scanner failed",
                exc_info=True,
                extra={
                    "scanner": self.name,
                    "pipeline": self.pipeline,
                    "error_type": type(e).__name__,
                },
            )
            return []

    def is_enabled(self) -> bool:
        return self.enabled


class ScannerRegistry:
    """Global scanner registry."""

    def __init__(self):
        self.scanners: Dict[str, Type[BaseScanner]] = {}

    def register(self, scanner_class: Type[BaseScanner]):
        """Register a scanner class with validation at registration time."""
        # Validate at registration time to fail fast
        if not hasattr(scanner_class, "name") or scanner_class.name is None:
            raise ValueError(f"{scanner_class.__name__} must define class attribute 'name'")
        if not hasattr(scanner_class, "pipeline") or scanner_class.pipeline is None:
            raise ValueError(f"{scanner_class.__name__} must define class attribute 'pipeline'")
        if not hasattr(scanner_class, "strategy") or scanner_class.strategy is None:
            raise ValueError(f"{scanner_class.__name__} must define class attribute 'strategy'")

        # Check for duplicate registration
        if scanner_class.name in self.scanners:
            logger.warning(f"Scanner '{scanner_class.name}' already registered, overwriting")

        self.scanners[scanner_class.name] = scanner_class
        logger.info(
            f"Registered scanner: {scanner_class.name} (pipeline: {scanner_class.pipeline})"
        )

    def get_scanners_by_pipeline(self, pipeline: str) -> List[Type[BaseScanner]]:
        return [sc for sc in self.scanners.values() if sc.pipeline == pipeline]

    def get_all_scanners(self) -> List[Type[BaseScanner]]:
        return list(self.scanners.values())


SCANNER_REGISTRY = ScannerRegistry()
