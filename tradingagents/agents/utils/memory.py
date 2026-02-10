import os
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from openai import OpenAI

from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class FinancialSituationMemory:
    def __init__(self, name, config):
        # Determine embedding backend URL
        # For Ollama, use the Ollama endpoint; otherwise default to OpenAI for embeddings
        if config.get("backend_url") == "http://localhost:11434/v1":
            self.embedding_backend = "http://localhost:11434/v1"
            self.embedding = "nomic-embed-text"
        else:
            # Always use OpenAI for embeddings, regardless of LLM provider
            self.embedding_backend = "https://api.openai.com/v1"
            self.embedding = "text-embedding-3-small"

        self.client = OpenAI(api_key=config.validate_key("openai_api_key", "OpenAI"))

        # Use persistent storage in project directory
        persist_directory = os.path.join(config.get("project_dir", "."), "memory_db")
        os.makedirs(persist_directory, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        try:
            self.situation_collection = self.chroma_client.get_collection(name=name)
        except Exception:
            self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        """Get OpenAI embedding for a text"""

        response = self.client.embeddings.create(model=self.embedding, input=text)
        return response.data[0].embedding

    def _batch_add(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: List[str] = None,
    ):
        """Internal helper to batch add documents to ChromaDB."""
        if not documents:
            return

        if ids is None:
            offset = self.situation_collection.count()
            ids = [str(offset + i) for i in range(len(documents))]

        self.situation_collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids,
        )

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""
        situations = []
        metadatas = []
        embeddings = []

        for situation, recommendation in situations_and_advice:
            situations.append(situation)
            metadatas.append({"recommendation": recommendation})
            embeddings.append(self.get_embedding(situation))

        self._batch_add(situations, metadatas, embeddings)

    def add_situations_with_metadata(
        self, situations_and_outcomes: List[Tuple[str, str, Dict[str, Any]]]
    ):
        """
        Add financial situations with enhanced metadata for learning system.

        Args:
            situations_and_outcomes: List of tuples (situation_text, recommendation, metadata)
                where metadata contains:
                - ticker: Stock symbol
                - analysis_date: Date of analysis (YYYY-MM-DD)
                - days_before_move: How many days before the major move (7 or 30)
                - move_pct: Percentage move that occurred
                - move_direction: "up" or "down"
                - agent_recommendation: What the agent recommended
                - was_correct: Boolean, whether recommendation matched outcome
                - structured_signals: Dict of signal features (optional)
                  - unusual_volume: bool
                  - analyst_sentiment: str (bullish/bearish/neutral)
                  - news_sentiment: str (positive/negative/neutral)
                  - short_interest: str (high/medium/low)
                  - insider_activity: str (buying/selling/none)
                  - etc.
        """
        situations = []
        metadatas = []
        embeddings = []

        for situation, recommendation, metadata in situations_and_outcomes:
            situations.append(situation)
            embeddings.append(self.get_embedding(situation))

            # Merge recommendation with metadata
            full_metadata = {"recommendation": recommendation}
            full_metadata.update(metadata)

            # Ensure all metadata values are strings, numbers, or booleans for ChromaDB
            full_metadata = self._sanitize_metadata(full_metadata)
            metadatas.append(full_metadata)

        self._batch_add(situations, metadatas, embeddings)

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata for ChromaDB compatibility.
        ChromaDB requires metadata values to be str, int, float, or bool.
        Nested dicts are flattened with dot notation.
        """
        sanitized = {}

        for key, value in metadata.items():
            if isinstance(value, dict):
                # Flatten nested dicts
                for nested_key, nested_value in value.items():
                    flat_key = f"{key}.{nested_key}"
                    if isinstance(nested_value, (str, int, float, bool, type(None))):
                        sanitized[flat_key] = nested_value if nested_value is not None else "none"
            elif isinstance(value, (str, int, float, bool, type(None))):
                sanitized[key] = value if value is not None else "none"
            else:
                # Convert other types to string
                sanitized[key] = str(value)

        return sanitized

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results

    def get_memories_hybrid(
        self,
        current_situation: str,
        signal_filters: Optional[Dict[str, Any]] = None,
        n_matches: int = 3,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: Filter by structured signals, then rank by embedding similarity.

        Args:
            current_situation: Text description of current market situation
            signal_filters: Dict of structured signals to filter by (e.g., {"unusual_volume": True})
                          Supports exact matches and can use dot notation for nested fields
                          e.g., {"structured_signals.unusual_volume": True}
            n_matches: Number of results to return
            min_similarity: Minimum similarity score (0-1) to include in results

        Returns:
            List of dicts with keys:
                - matched_situation: Historical situation text
                - recommendation: What was recommended
                - similarity_score: Embedding similarity (0-1)
                - metadata: Full metadata including outcome, signals, etc.
        """
        query_embedding = self.get_embedding(current_situation)

        # Build where clause for filtering
        where_clause = None
        if signal_filters:
            where_clause = {}
            for key, value in signal_filters.items():
                where_clause[key] = value

        # Query ChromaDB with optional filtering
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_matches * 3, 100),  # Get more results for filtering
            "include": ["metadatas", "documents", "distances"],
        }

        if where_clause:
            query_params["where"] = where_clause

        results = self.situation_collection.query(**query_params)

        # Process and filter results
        matched_results = []
        for i in range(len(results["documents"][0])):
            similarity_score = 1 - results["distances"][0][i]

            # Apply similarity threshold
            if similarity_score < min_similarity:
                continue

            metadata = results["metadatas"][0][i]

            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": metadata.get("recommendation", ""),
                    "similarity_score": similarity_score,
                    "metadata": metadata,
                    # Extract key fields for convenience
                    "ticker": metadata.get("ticker", ""),
                    "move_pct": metadata.get("move_pct", 0),
                    "move_direction": metadata.get("move_direction", ""),
                    "was_correct": metadata.get("was_correct", False),
                    "days_before_move": metadata.get("days_before_move", 0),
                }
            )

        # Return top n_matches
        return matched_results[:n_matches]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the memory bank.

        Returns:
            Dict with keys:
                - total_memories: Total number of stored memories
                - accuracy_rate: % of memories where was_correct=True
                - avg_move_pct: Average percentage move in stored outcomes
                - signal_distribution: Count of different signal patterns
        """
        total_count = self.situation_collection.count()

        if total_count == 0:
            return {
                "total_memories": 0,
                "accuracy_rate": 0.0,
                "avg_move_pct": 0.0,
                "signal_distribution": {},
            }

        # Get all memories
        all_results = self.situation_collection.get(include=["metadatas"])

        metadatas = all_results["metadatas"]

        # Calculate statistics
        correct_count = sum(1 for m in metadatas if m.get("was_correct") == True)
        accuracy_rate = (correct_count / total_count * 100) if total_count > 0 else 0

        move_pcts = [m.get("move_pct", 0) for m in metadatas if "move_pct" in m]
        avg_move_pct = sum(move_pcts) / len(move_pcts) if move_pcts else 0

        # Count signal patterns
        signal_distribution = {}
        for metadata in metadatas:
            for key, value in metadata.items():
                if key.startswith("structured_signals."):
                    signal_name = key.replace("structured_signals.", "")
                    if signal_name not in signal_distribution:
                        signal_distribution[signal_name] = {}
                    if value not in signal_distribution[signal_name]:
                        signal_distribution[signal_name][value] = 0
                    signal_distribution[signal_name][value] += 1

        return {
            "total_memories": total_count,
            "accuracy_rate": accuracy_rate,
            "avg_move_pct": avg_move_pct,
            "signal_distribution": signal_distribution,
        }


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            logger.info(f"Match {i}:")
            logger.info(f"Similarity Score: {rec['similarity_score']:.2f}")
            logger.info(f"Matched Situation: {rec['matched_situation']}")
            logger.info(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        logger.error(f"Error during recommendation: {str(e)}")
