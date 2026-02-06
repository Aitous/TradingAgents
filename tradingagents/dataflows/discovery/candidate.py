from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Candidate:
    """Lightweight candidate wrapper for discovery flow."""

    ticker: str
    source: str = ""
    priority: str = "unknown"
    context: str = ""
    allow_invalid: bool = False
    all_sources: List[str] = field(default_factory=list)
    context_details: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candidate":
        known_keys = {
            "ticker",
            "source",
            "priority",
            "context",
            "allow_invalid",
            "all_sources",
            "context_details",
            "sources",
            "contexts",
        }
        extras = {k: v for k, v in data.items() if k not in known_keys}

        candidate = cls(
            ticker=(data.get("ticker") or "").upper().strip(),
            source=data.get("source", "") or "",
            priority=data.get("priority", "unknown") or "unknown",
            context=data.get("context", "") or "",
            allow_invalid=bool(data.get("allow_invalid", False)),
            all_sources=list(data.get("all_sources") or data.get("sources") or []),
            context_details=list(data.get("context_details") or data.get("contexts") or []),
            extras=extras,
        )
        candidate.normalize()
        return candidate

    def normalize(self) -> None:
        """Ensure sources/context lists are populated and deduped."""
        if not self.all_sources and self.source:
            self.all_sources = [self.source]
        if not self.context_details and self.context:
            self.context_details = [self.context]

        self.all_sources = list(dict.fromkeys([s for s in self.all_sources if s]))
        self.context_details = list(dict.fromkeys([c for c in self.context_details if c]))

        if not self.source and self.all_sources:
            self.source = self.all_sources[0]
        if not self.context and self.context_details:
            self.context = self.context_details[0]

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.extras)
        data.update(
            {
                "ticker": self.ticker,
                "source": self.source,
                "priority": self.priority,
                "context": self.context,
                "allow_invalid": self.allow_invalid,
                "all_sources": self.all_sources,
                "context_details": self.context_details,
            }
        )
        return data
