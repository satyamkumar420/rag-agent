"""
Query Processor Module

This module is responsible for processing user queries and converting
them to vector embeddings for retrieval.

Technologies: Gemini Embedding v3, LangChain, Pinecone
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class QueryProcessor:
    """
    Processes user queries and converts them to vector embeddings.

    Features:
    - Query preprocessing and normalization
    - Query embedding generation
    - Context retrieval from vector database
    - Query expansion and caching
    - Metadata filtering and ranking
    """

    def __init__(
        self, embedding_generator, vector_db, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the QueryProcessor with dependencies.

        Args:
            embedding_generator: Instance of EmbeddingGenerator
            vector_db: Instance of VectorDB
            config: Configuration dictionary with processing parameters
        """
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.top_k = self.config.get("top_k", 5)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.max_context_length = self.config.get("max_context_length", 4000)
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour

        # Query cache and history
        self.query_cache = {}
        self.query_history = []

        self.logger.info("QueryProcessor initialized with advanced features")

    def process_query(
        self, query: str, filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query and retrieve relevant context.

        Args:
            query: User query string
            filter: Optional metadata filter for search

        Returns:
            Dictionary containing query, retrieved context, and metadata
        """
        if not query or not query.strip():
            return {
                "query": query,
                "context": [],
                "total_results": 0,
                "error": "Empty query provided",
            }

        self.logger.info(f"Processing query: {query[:100]}...")
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, filter)
            if self.enable_caching and cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                if self._is_cache_valid(cached_result["timestamp"]):
                    self.logger.info("  Returning cached result")
                    cached_result["from_cache"] = True
                    return cached_result

            # Preprocess the query
            processed_query = self._preprocess_query(query)
            expanded_queries = self._expand_query(processed_query)

            # Generate embeddings for all query variations
            all_results = []
            for q in expanded_queries:
                query_embedding = self.embedding_generator.generate_query_embedding(q)

                if query_embedding:
                    # Search for similar vectors
                    search_results = self.vector_db.search(
                        query_embedding=query_embedding,
                        top_k=self.top_k * 2,  # Get more results for better filtering
                        filter=filter,
                    )
                    all_results.extend(search_results)

            # Deduplicate and rank results
            unique_results = self._deduplicate_results(all_results)
            ranked_results = self._rank_results(unique_results, query)

            # Filter results by similarity threshold
            filtered_results = [
                result
                for result in ranked_results[: self.top_k]
                if result.get("score", 0) >= self.similarity_threshold
            ]

            # Extract and format context
            context = self._extract_context(filtered_results)

            # Prepare result
            result = {
                "query": query,
                "processed_query": processed_query,
                "expanded_queries": expanded_queries,
                "context": context,
                "total_results": len(filtered_results),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now(),
                "from_cache": False,
            }

            # Cache the result
            if self.enable_caching:
                self.query_cache[cache_key] = result.copy()

            # Add to query history
            self._add_to_history(query, len(filtered_results))

            self.logger.info(f"Query processed in {result['processing_time']:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Error processing query: {str(e)}")
            return {
                "query": query,
                "context": [],
                "total_results": 0,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query for better embedding generation.

        Args:
            query: Raw query string

        Returns:
            Preprocessed query string
        """
        # Remove extra whitespace
        query = " ".join(query.split())

        # Remove special characters that might interfere
        import re

        query = re.sub(r"[^\w\s\-\?\!]", " ", query)

        # Normalize question words
        question_words = {
            "whats": "what is",
            "hows": "how is",
            "wheres": "where is",
            "whos": "who is",
            "whens": "when is",
        }

        for abbrev, full in question_words.items():
            query = query.replace(abbrev, full)

        return query.strip()

    def _expand_query(self, query: str) -> List[str]:
        """
        Expand the query with variations for better retrieval.

        Args:
            query: Preprocessed query

        Returns:
            List of query variations
        """
        expanded = [query]

        # Add question variations
        if not any(
            q in query.lower() for q in ["what", "how", "why", "when", "where", "who"]
        ):
            expanded.append(f"what is {query}")
            expanded.append(f"how does {query} work")

        # Add definition variation
        if "definition" not in query.lower() and "define" not in query.lower():
            expanded.append(f"{query} definition")

        # Add example variation
        if "example" not in query.lower():
            expanded.append(f"{query} examples")

        return expanded[:3]  # Limit to 3 variations

    def _deduplicate_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on content similarity.

        Args:
            results: List of search results

        Returns:
            Deduplicated results
        """
        seen_ids = set()
        unique_results = []

        for result in results:
            result_id = result.get("id")
            if result_id and result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)

        return unique_results

    def _rank_results(
        self, results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """
        Rank results based on multiple factors.

        Args:
            results: List of search results
            query: Original query

        Returns:
            Ranked results
        """
        query_words = set(query.lower().split())

        for result in results:
            # Base score from similarity
            base_score = result.get("score", 0.0)

            # Boost score based on text relevance
            text = result.get("metadata", {}).get("text", "").lower()
            text_words = set(text.split())
            word_overlap = len(query_words.intersection(text_words))
            relevance_boost = word_overlap / max(len(query_words), 1) * 0.1

            # Boost score based on source type
            source = result.get("metadata", {}).get("source", "")
            source_boost = 0.0
            if source.endswith(".pdf"):
                source_boost = 0.05  # PDFs often contain structured info
            elif "http" in source:
                source_boost = 0.02  # Web content

            # Calculate final score
            final_score = base_score + relevance_boost + source_boost
            result["final_score"] = min(final_score, 1.0)

        # Sort by final score
        return sorted(results, key=lambda x: x.get("final_score", 0), reverse=True)

    def _extract_context(
        self, search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract and format context from search results.

        Args:
            search_results: List of search results from vector database

        Returns:
            List of formatted context items
        """
        context = []
        total_length = 0

        for result in search_results:
            # Extract text content from metadata
            text = result.get("metadata", {}).get("text", "")

            # Check if adding this context would exceed the limit
            if total_length + len(text) > self.max_context_length and context:
                break

            # Format context item with enhanced metadata
            context_item = {
                "text": text,
                "score": result.get("score", 0),
                "final_score": result.get("final_score", result.get("score", 0)),
                "source": result.get("metadata", {}).get("source", "unknown"),
                "chunk_id": result.get("id", ""),
                "metadata": result.get("metadata", {}),
                "relevance_rank": len(context) + 1,
            }

            context.append(context_item)
            total_length += len(text)

        self.logger.info(
            f"Extracted {len(context)} context items (total length: {total_length})"
        )
        return context

    def _generate_cache_key(self, query: str, filter: Optional[Dict[str, Any]]) -> str:
        """Generate a cache key for the query."""
        import hashlib

        filter_str = str(sorted(filter.items())) if filter else ""
        cache_string = f"{query.lower().strip()}{filter_str}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached result is still valid."""
        return datetime.now() - timestamp < timedelta(seconds=self.cache_ttl)

    def _add_to_history(self, query: str, result_count: int):
        """Add query to history for analytics."""
        self.query_history.append(
            {
                "query": query,
                "timestamp": datetime.now(),
                "result_count": result_count,
            }
        )

        # Keep only last 100 queries
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]

    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """
        Generate query suggestions based on partial input and history.

        Args:
            partial_query: Partial query string

        Returns:
            List of suggested queries
        """
        suggestions = []

        # Add suggestions from query history
        for hist_item in reversed(self.query_history[-20:]):  # Last 20 queries
            hist_query = hist_item["query"]
            if (
                partial_query.lower() in hist_query.lower()
                and hist_query not in suggestions
            ):
                suggestions.append(hist_query)

        # Add template-based suggestions
        if len(suggestions) < 3:
            templates = [
                f"What is {partial_query}?",
                f"How does {partial_query} work?",
                f"Examples of {partial_query}",
                f"{partial_query} definition",
                f"{partial_query} best practices",
            ]

            for template in templates:
                if template not in suggestions:
                    suggestions.append(template)
                if len(suggestions) >= 5:
                    break

        return suggestions[:5]

    def get_query_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about query patterns.

        Returns:
            Dictionary with query analytics
        """
        if not self.query_history:
            return {"total_queries": 0, "cache_hit_rate": 0.0}

        total_queries = len(self.query_history)
        recent_queries = [q["query"] for q in self.query_history[-10:]]

        # Calculate average results per query
        avg_results = sum(q["result_count"] for q in self.query_history) / total_queries

        # Most common query patterns
        query_words = []
        for q in self.query_history:
            query_words.extend(q["query"].lower().split())

        from collections import Counter

        common_words = Counter(query_words).most_common(5)

        return {
            "total_queries": total_queries,
            "average_results_per_query": round(avg_results, 2),
            "recent_queries": recent_queries,
            "common_query_words": common_words,
            "cache_size": len(self.query_cache),
        }

    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()
        self.logger.info("Query cache cleared")

    def clear_history(self):
        """Clear the query history."""
        self.query_history.clear()
        self.logger.info("Query history cleared")
