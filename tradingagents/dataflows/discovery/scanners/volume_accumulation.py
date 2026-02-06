"""Volume accumulation and compression scanner."""
from typing import Any, Dict, List

from tradingagents.dataflows.discovery.scanner_registry import BaseScanner, SCANNER_REGISTRY
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.tools.executor import execute_tool


class VolumeAccumulationScanner(BaseScanner):
    """Scan for unusual volume accumulation patterns."""

    name = "volume_accumulation"
    pipeline = "momentum"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.unusual_volume_multiple = self.scanner_config.get("unusual_volume_multiple", 2.0)
        self.volume_cache_key = self.scanner_config.get("volume_cache_key", "default")

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        print(f"   📊 Scanning volume accumulation...")

        try:
            # Use volume scanner tool
            result = execute_tool(
                "get_unusual_volume",
                min_volume_multiple=self.unusual_volume_multiple,
                top_n=self.limit
            )

            if not result:
                print(f"      Found 0 volume accumulation candidates")
                return []

            candidates = []

            # Handle different result formats
            if isinstance(result, str):
                # Parse markdown/text result
                candidates = self._parse_text_result(result)
            elif isinstance(result, list):
                # Structured result
                for item in result[:self.limit]:
                    ticker = item.get("ticker", "").upper()
                    if not ticker:
                        continue

                    volume_ratio = item.get("volume_ratio", 0)
                    avg_volume = item.get("avg_volume", 0)

                    candidates.append({
                        "ticker": ticker,
                        "source": self.name,
                        "context": f"Unusual volume: {volume_ratio:.1f}x average ({avg_volume:,})",
                        "priority": Priority.MEDIUM.value if volume_ratio < 3.0 else Priority.HIGH.value,
                        "strategy": "volume_accumulation",
                    })
            elif isinstance(result, dict):
                # Dict with tickers list
                for ticker in result.get("tickers", [])[:self.limit]:
                    candidates.append({
                        "ticker": ticker.upper(),
                        "source": self.name,
                        "context": f"Unusual volume accumulation",
                        "priority": Priority.MEDIUM.value,
                        "strategy": "volume_accumulation",
                    })

            print(f"      Found {len(candidates)} volume accumulation candidates")
            return candidates

        except Exception as e:
            print(f"      ⚠️  Volume accumulation failed: {e}")
            return []

    def _parse_text_result(self, text: str) -> List[Dict[str, Any]]:
        """Parse tickers from text result."""
        from tradingagents.dataflows.discovery.common_utils import extract_tickers_from_text

        candidates = []
        tickers = extract_tickers_from_text(text)

        for ticker in tickers[:self.limit]:
            candidates.append({
                "ticker": ticker,
                "source": self.name,
                "context": "Unusual volume detected",
                "priority": Priority.MEDIUM.value,
                "strategy": "volume_accumulation",
            })

        return candidates


SCANNER_REGISTRY.register(VolumeAccumulationScanner)
