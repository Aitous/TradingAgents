"""
Ticker Semantic Database
------------------------
Creates and maintains a database of ticker descriptions with embeddings
for semantic matching against news events.

This enables news-driven discovery by finding tickers semantically related
to breaking news, rather than waiting for social media buzz or price action.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from tradingagents.dataflows.y_finance import get_ticker_info
from tradingagents.utils.logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class TickerSemanticDB:
    """Manages ticker descriptions and embeddings for semantic search."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ticker semantic database.

        Args:
            config: Configuration dict with:
                - project_dir: Base directory for storage
                - use_openai_embeddings: If True, use OpenAI; else use local HF model
                - embedding_model: Model name (default: text-embedding-3-small)
        """
        self.config = config
        self.use_openai = config.get("use_openai_embeddings", True)

        # Setup embedding backend
        if self.use_openai:
            self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.openai_client = OpenAI(api_key=openai_api_key)
            self.embedding_dim = 1536  # OpenAI text-embedding-3-small dimension
        else:
            # TODO: Add local HuggingFace model support
            # Use sentence-transformers with a good MTEB-ranked model
            from sentence_transformers import SentenceTransformer

            self.embedding_model = config.get("embedding_model", "BAAI/bge-small-en-v1.5")
            self.local_model = SentenceTransformer(self.embedding_model)
            self.embedding_dim = self.local_model.get_sentence_embedding_dimension()

        # Setup ChromaDB for persistent storage
        project_dir = config.get("project_dir", ".")
        embedding_model_safe = self.embedding_model.replace("/", "_").replace(" ", "_")
        db_dir = os.path.join(project_dir, "ticker_semantic_db", embedding_model_safe)
        os.makedirs(db_dir, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=db_dir)

        # Get or create collection
        collection_name = "ticker_descriptions"
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(f"Loaded existing ticker database: {self.collection.count()} tickers")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Ticker descriptions with metadata for semantic search"},
            )
            logger.info("Created new ticker database collection")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using configured backend."""
        if self.use_openai:
            response = self.openai_client.embeddings.create(model=self.embedding_model, input=text)
            return response.data[0].embedding
        else:
            # Local HuggingFace model
            embedding = self.local_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

    def fetch_ticker_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch ticker information from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with ticker metadata or None if fetch fails
        """
        try:
            info = get_ticker_info(symbol)

            # Extract relevant fields
            description = info.get("longBusinessSummary", "")
            if not description:
                # Fallback to shorter description if available
                description = info.get("description", f"{symbol} - No description available")

            # Build metadata dict
            ticker_data = {
                "symbol": symbol.upper(),
                "name": info.get("longName", info.get("shortName", symbol)),
                "description": description,
                "industry": info.get("industry", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "revenue": info.get("totalRevenue", 0),
                "country": info.get("country", "US"),
                "website": info.get("website", ""),
                "employees": info.get("fullTimeEmployees", 0),
                "last_updated": datetime.now().isoformat(),
            }

            return ticker_data

        except Exception as e:
            logger.warning(f"Error fetching {symbol}: {e}")
            return None

    def add_ticker(self, symbol: str, force_refresh: bool = False) -> bool:
        """
        Add a single ticker to the database.

        Args:
            symbol: Stock ticker symbol
            force_refresh: If True, refresh even if ticker exists

        Returns:
            True if added successfully, False otherwise
        """
        # Check if already exists
        if not force_refresh:
            try:
                existing = self.collection.get(ids=[symbol.upper()])
                if existing and existing["ids"]:
                    return True  # Already exists
            except Exception:
                pass

        # Fetch ticker info
        ticker_data = self.fetch_ticker_info(symbol)
        if not ticker_data:
            return False

        # Generate embedding from description
        try:
            embedding = self.get_embedding(ticker_data["description"])
        except Exception as e:
            logger.error(f"Error generating embedding for {symbol}: {e}")
            return False

        # Store in ChromaDB
        try:
            # Store description as document, metadata as metadata, embedding as embedding
            self.collection.upsert(
                ids=[symbol.upper()],
                documents=[ticker_data["description"]],
                embeddings=[embedding],
                metadatas=[
                    {
                        "symbol": ticker_data["symbol"],
                        "name": ticker_data["name"],
                        "industry": ticker_data["industry"],
                        "sector": ticker_data["sector"],
                        "market_cap": ticker_data["market_cap"],
                        "revenue": ticker_data["revenue"],
                        "country": ticker_data["country"],
                        "website": ticker_data["website"],
                        "employees": ticker_data["employees"],
                        "last_updated": ticker_data["last_updated"],
                    }
                ],
            )
            return True
        except Exception as e:
            logger.error(f"Error storing {symbol}: {e}")
            return False

    def build_database(
        self,
        ticker_file: str,
        max_tickers: Optional[int] = None,
        skip_existing: bool = True,
        batch_size: int = 100,
    ):
        """
        Build the ticker database from a file.

        Args:
            ticker_file: Path to file with ticker symbols (one per line)
            max_tickers: Maximum number of tickers to process (None = all)
            skip_existing: If True, skip tickers already in DB
            batch_size: Number of tickers to process before showing progress
        """
        # Read ticker file
        with open(ticker_file, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip()]

        if max_tickers:
            tickers = tickers[:max_tickers]

        logger.info("Building ticker semantic database...")
        logger.info(f"Source: {ticker_file}")
        logger.info(f"Total tickers: {len(tickers)}")
        logger.info(f"Embedding model: {self.embedding_model}")

        # Get existing tickers if skipping
        existing_tickers = set()
        if skip_existing:
            try:
                existing = self.collection.get(include=[])
                existing_tickers = set(existing["ids"])
                logger.info(f"Existing tickers in DB: {len(existing_tickers)}")
            except Exception:
                pass

        # Process tickers
        success_count = 0
        skip_count = 0
        fail_count = 0

        for i, symbol in enumerate(tqdm(tickers, desc="Processing tickers")):
            # Skip if exists
            if skip_existing and symbol in existing_tickers:
                skip_count += 1
                continue

            # Add ticker
            if self.add_ticker(symbol, force_refresh=not skip_existing):
                success_count += 1
            else:
                fail_count += 1

        logger.info("Database build complete!")
        logger.info(f"Success: {success_count}")
        logger.info(f"Skipped: {skip_count}")
        logger.info(f"Failed: {fail_count}")
        logger.info(f"Total in DB: {self.collection.count()}")

    def search_by_text(
        self, query_text: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for tickers semantically related to query text.

        Args:
            query_text: Text to search for (e.g., news summary)
            top_k: Number of top matches to return
            filters: Optional metadata filters (e.g., {"sector": "Technology"})

        Returns:
            List of ticker matches with metadata and similarity scores
        """
        # Generate embedding for query
        query_embedding = self.get_embedding(query_text)

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,  # Apply metadata filters if provided
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        matches = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 / (1 + distance)
            match = {
                "symbol": results["ids"][0][i],
                "description": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": similarity,  # Normalize distance to (0, 1]
            }
            matches.append(match)

        return matches

    def get_ticker_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stored information for a specific ticker."""
        try:
            result = self.collection.get(ids=[symbol.upper()], include=["documents", "metadatas"])

            if not result["ids"]:
                return None

            return {
                "symbol": result["ids"][0],
                "description": result["documents"][0],
                "metadata": result["metadatas"][0],
            }
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            count = self.collection.count()

            # Get sector breakdown
            all_data = self.collection.get(include=["metadatas"])
            sectors = {}
            industries = {}

            for metadata in all_data["metadatas"]:
                sector = metadata.get("sector", "Unknown")
                industry = metadata.get("industry", "Unknown")
                sectors[sector] = sectors.get(sector, 0) + 1
                industries[industry] = industries.get(industry, 0) + 1

            return {
                "total_tickers": count,
                "sectors": sectors,
                "industries": industries,
                "embedding_model": self.embedding_model,
                "embedding_dimension": self.embedding_dim,
            }
        except Exception as e:
            return {"error": str(e)}


def main():
    """CLI for building/managing the ticker database."""
    import argparse

    parser = argparse.ArgumentParser(description="Build ticker semantic database")
    parser.add_argument("--ticker-file", default="data/tickers.txt", help="Path to ticker file")
    parser.add_argument(
        "--max-tickers", type=int, default=None, help="Maximum tickers to process (default: all)"
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Use local HuggingFace embeddings instead of OpenAI",
    )
    parser.add_argument(
        "--force-refresh", action="store_true", help="Refresh all tickers even if they exist"
    )
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--search", type=str, help="Search for tickers by text query")

    args = parser.parse_args()

    # Load config
    from tradingagents.default_config import DEFAULT_CONFIG

    config = {
        "project_dir": DEFAULT_CONFIG["project_dir"],
        "use_openai_embeddings": not args.use_local,
    }

    # Initialize database
    db = TickerSemanticDB(config)

    # Execute command
    if args.stats:
        stats = db.get_stats()
        logger.info("📊 Database Statistics:")
        logger.info(json.dumps(stats, indent=2))

    elif args.search:
        logger.info(f"🔍 Searching for: {args.search}")
        matches = db.search_by_text(args.search, top_k=10)
        logger.info("Top matches:")
        for i, match in enumerate(matches, 1):
            logger.info(f"{i}. {match['symbol']} - {match['metadata']['name']}")
            logger.debug(f"  Sector: {match['metadata']['sector']}")
            logger.debug(f"  Similarity: {match['similarity_score']:.3f}")
            logger.debug(f"  Description: {match['description'][:150]}...")

    else:
        # Build database
        db.build_database(
            ticker_file=args.ticker_file,
            max_tickers=args.max_tickers,
            skip_existing=not args.force_refresh,
        )


if __name__ == "__main__":
    main()
