"""
Optimized Query Processor with Rate Limiting and Better Error Handling
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class OptimizedQueryProcessor:
    """
    Optimized QueryProcessor with rate limiting and better error handling
    """

    def __init__(
        self, embedding_generator, vector_db, config: Optional[Dict[str, Any]] = None
    ):
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Optimized configuration settings
        self.top_k = self.config.get("top_k", 10)  # Increased from 5
        self.similarity_threshold = self.config.get(
            "similarity_threshold", 0.4
        )  # Lowered from 0.7
        self.max_context_length = self.config.get(
            "max_context_length", 8000
        )  # Increased
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_ttl = self.config.get("cache_ttl", 7200)  # 2 hours

        # Rate limiting settings
        self.last_api_call = 0
        self.min_api_interval = 1.0  # Minimum 1 second between API calls
        self.max_retries = 3
        self.retry_delay = 2.0

        # Query cache and history
        self.query_cache = {}
        self.query_history = []

        self.logger.info("OptimizedQueryProcessor initialized")

    def process_query(
        self, query: str, filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process query with optimized rate limiting and error handling
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
                    self.logger.info("Returning cached result")
                    cached_result["from_cache"] = True
                    return cached_result

            # Rate limiting protection
            self._enforce_rate_limit()

            # Generate query embedding with retry logic
            query_embedding = self._generate_embedding_with_retry(query)

            if not query_embedding:
                return {
                    "query": query,
                    "context": [],
                    "total_results": 0,
                    "error": "Failed to generate query embedding",
                }

            # Search for similar vectors with increased top_k
            search_results = self.vector_db.search(
                query_embedding=query_embedding,
                top_k=self.top_k * 2,  # Get more results for better filtering
                filter=filter,
                include_metadata=True,
            )

            if not search_results:
                self.logger.warning("No search results returned from vector database")
                return {
                    "query": query,
                    "context": [],
                    "total_results": 0,
                    "error": "No similar documents found",
                }

            # Apply optimized filtering
            filtered_results = self._apply_smart_filtering(search_results, query)

            # Extract and format context with better error handling
            context = self._extract_context_safely(filtered_results)

            # Prepare result
            result = {
                "query": query,
                "context": context,
                "total_results": len(filtered_results),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now(),
                "from_cache": False,
                "similarity_scores": [r.get("score", 0) for r in filtered_results[:5]],
            }

            # Cache the result
            if self.enable_caching:
                self.query_cache[cache_key] = result.copy()

            self.logger.info(
                f"Query processed in {result['processing_time']:.2f}s, {len(context)} context items"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "context": [],
                "total_results": 0,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def _enforce_rate_limit(self):
        """Enforce rate limiting between API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call

        if time_since_last_call < self.min_api_interval:
            sleep_time = self.min_api_interval - time_since_last_call
            self.logger.info(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

        self.last_api_call = time.time()

    def _generate_embedding_with_retry(self, query: str) -> List[float]:
        """Generate embedding with retry logic and rate limiting"""
        for attempt in range(self.max_retries):
            try:
                self._enforce_rate_limit()
                embedding = self.embedding_generator.generate_query_embedding(query)

                if embedding:
                    return embedding
                else:
                    self.logger.warning(
                        f"Attempt {attempt + 1}: Empty embedding returned"
                    )

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                if "429" in str(e) or "quota" in str(e).lower():
                    # Rate limit hit - wait longer
                    wait_time = self.retry_delay * (2**attempt)
                    self.logger.info(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        self.logger.error("All embedding generation attempts failed")
        return []

    def _apply_smart_filtering(
        self, search_results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Apply smart filtering with adaptive threshold"""
        if not search_results:
            return []

        # Get score statistics
        scores = [r.get("score", 0) for r in search_results]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        # Adaptive threshold: use lower threshold if max score is low
        adaptive_threshold = min(self.similarity_threshold, max_score * 0.8)

        self.logger.info(
            f"Score stats - Max: {max_score:.3f}, Avg: {avg_score:.3f}, Threshold: {adaptive_threshold:.3f}"
        )

        # Filter results
        filtered = [
            result
            for result in search_results[: self.top_k]
            if result.get("score", 0) >= adaptive_threshold
        ]

        # If no results pass threshold, return top 3 anyway
        if not filtered and search_results:
            self.logger.warning(
                f"No results above threshold {adaptive_threshold:.3f}, returning top 3"
            )
            filtered = search_results[:3]

        return filtered

    def _extract_context_safely(
        self, search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract context with better error handling"""
        context = []
        total_length = 0

        for i, result in enumerate(search_results):
            try:
                # Multiple ways to extract text content
                text = ""
                metadata = result.get("metadata", {})

                # Try different text fields
                for field in ["text", "content", "content_preview", "description"]:
                    if field in metadata and metadata[field]:
                        text = str(metadata[field])
                        break

                if not text:
                    self.logger.warning(f"No text content found in result {i}")
                    continue

                # Check length limit
                if total_length + len(text) > self.max_context_length and context:
                    break

                # Create context item
                context_item = {
                    "text": text,
                    "score": result.get("score", 0),
                    "source": metadata.get("source", f"Document {i+1}"),
                    "chunk_id": result.get("id", ""),
                    "metadata": metadata,
                    "relevance_rank": len(context) + 1,
                }

                context.append(context_item)
                total_length += len(text)

            except Exception as e:
                self.logger.warning(f"Error extracting context from result {i}: {e}")
                continue

        self.logger.info(
            f"Extracted {len(context)} context items (total length: {total_length})"
        )
        return context

    def _generate_cache_key(self, query: str, filter: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for query"""
        import hashlib

        filter_str = str(sorted(filter.items())) if filter else ""
        cache_string = f"{query.lower().strip()}{filter_str}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached result is still valid"""
        return datetime.now() - timestamp < timedelta(seconds=self.cache_ttl)
