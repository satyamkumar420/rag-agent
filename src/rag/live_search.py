"""
Live Search Processor using Tavily Python Client.
Provides real-time web search capabilities for the RAG system.
"""

import logging
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LiveSearchProcessor:
    """Handles live web search using Tavily Python Client."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LiveSearchProcessor.

        Args:
            config: Configuration dictionary containing live search settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Search configuration
        self.enabled = self.config.get("enabled", False)
        self.max_results = self.config.get("max_results", 5)
        self.search_depth = self.config.get("search_depth", "basic")
        self.include_answer = self.config.get("include_answer", True)
        self.include_raw_content = self.config.get("include_raw_content", False)
        self.include_images = self.config.get("include_images", False)
        self.topic = self.config.get("topic", "general")
        self.enable_caching = self.config.get("enable_caching", True)

        # Search cache and analytics
        self.search_cache = {}
        self.search_history = []

        # Initialize Tavily client
        self.tavily_client = None
        self._initialize_client()

        self.logger.info(f"LiveSearchProcessor initialized - Enabled: {self.enabled}")

    def _initialize_client(self):
        """Initialize the Tavily client."""
        try:
            # Get API key from environment variable
            api_key = os.getenv("TAVILY_API_KEY")

            if not api_key:
                self.logger.warning("TAVILY_API_KEY not found in environment variables")
                self.enabled = False
                return

            # Import and initialize Tavily client
            from tavily import TavilyClient

            self.tavily_client = TavilyClient(api_key=api_key)

            # ✅ Auto-enable if client initializes successfully and no explicit config
            if self.tavily_client and not self.config.get(
                "enabled_explicitly_set", False
            ):
                self.enabled = True
                self.logger.info(
                    "Tavily client initialized successfully - Auto-enabled live search"
                )
            else:
                self.logger.info("Tavily client initialized successfully")

        except ImportError:
            self.logger.error(
                "tavily-python package not installed. Install with: pip install tavily-python"
            )
            self.enabled = False
        except Exception as e:
            self.logger.error(f"Failed to initialize Tavily client: {str(e)}")
            self.enabled = False

    def is_enabled(self) -> bool:
        """Check if live search is enabled."""
        return self.enabled and self.tavily_client is not None

    def search_web(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: Optional[str] = None,
        time_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform live web search using Tavily API.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            search_depth: Search depth ('basic' or 'advanced')
            time_range: Time range for search results

        Returns:
            Dictionary containing search results and metadata
        """
        if not query or not query.strip():
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": "Empty query provided",
                "source": "live_search",
            }

        if not self.is_enabled():
            self.logger.warning("Live search is disabled or client not initialized")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": "Live search is disabled or Tavily client not initialized",
                "source": "live_search",
            }

        self.logger.info(f"Performing live search: {query[:100]}...")
        start_time = time.time()

        try:
            # Use provided parameters or defaults
            search_max_results = max_results or self.max_results
            search_depth_param = search_depth or self.search_depth

            # Check cache first
            cache_key = self._generate_cache_key(
                query, search_max_results, search_depth_param
            )
            if self.enable_caching and cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                if self._is_cache_valid(cached_result["timestamp"]):
                    self.logger.info("Returning cached search result")
                    cached_result["from_cache"] = True
                    return cached_result

            # Prepare search parameters
            search_params = {
                "query": query,
                "max_results": min(search_max_results, 20),  # Tavily limit
                "search_depth": search_depth_param,
                "include_answer": self.include_answer,
                "include_raw_content": self.include_raw_content,
                "include_images": self.include_images,
                "topic": self.topic,
            }

            # Add time_range if provided
            if time_range:
                search_params["time_range"] = time_range

            # Perform the search
            response = self.tavily_client.search(**search_params)

            # Process and format results
            processed_results = self._process_search_results(
                response.get("results", []), query
            )

            # Prepare final result
            result = {
                "query": query,
                "results": processed_results,
                "total_results": len(processed_results),
                "answer": response.get("answer"),
                "images": response.get("images", []),
                "follow_up_questions": response.get("follow_up_questions", []),
                "search_params": {
                    "max_results": search_max_results,
                    "search_depth": search_depth_param,
                    "time_range": time_range,
                },
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now(),
                "source": "live_search",
                "from_cache": False,
                "search_metadata": {
                    "source": "tavily",
                    "timestamp": datetime.now().isoformat(),
                    "results_count": len(processed_results),
                    "search_depth": search_depth_param,
                    "max_results": search_max_results,
                    "response_time": response.get("response_time"),
                },
            }

            # Cache the result
            if self.enable_caching:
                self.search_cache[cache_key] = result.copy()

            # Add to search history
            self._add_to_history(query, len(processed_results))

            self.logger.info(
                f"Live search completed in {result['processing_time']:.2f}s"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error in live search: {str(e)}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "source": "live_search",
            }

    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Perform a live web search using Tavily API.

        Args:
            query: Search query string
            **kwargs: Additional search parameters

        Returns:
            Dictionary containing search results
        """
        return self.search_web(query, **kwargs)

    def _process_search_results(
        self, raw_results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """
        Process and format raw search results from Tavily.

        Args:
            raw_results: Raw results from Tavily API
            query: Original search query

        Returns:
            Processed and formatted results
        """
        processed_results = []
        query_words = set(query.lower().split())

        for i, result in enumerate(raw_results):
            try:
                # Extract key information
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")
                raw_content = result.get("raw_content", "")
                score = result.get("score", 0.0)

                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(
                    title, content, query_words, score
                )

                # Format result
                formatted_result = {
                    "title": title,
                    "url": url,
                    "content": content[:500] + "..." if len(content) > 500 else content,
                    "raw_content": raw_content if self.include_raw_content else "",
                    "score": score,
                    "relevance_score": relevance_score,
                    "rank": i + 1,
                    "source": "web_search",
                    "search_engine": "tavily",
                    "published_date": result.get("published_date"),
                    "metadata": {
                        "title": title,
                        "url": url,
                        "content_length": len(content),
                        "has_raw_content": bool(raw_content),
                        "search_rank": i + 1,
                    },
                }

                processed_results.append(formatted_result)

            except Exception as e:
                self.logger.warning(f"Error processing search result {i}: {str(e)}")
                continue

        # Sort by relevance score
        processed_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return processed_results

    def _calculate_relevance_score(
        self, title: str, content: str, query_words: set, base_score: float
    ) -> float:
        """
        Calculate relevance score for search results.

        Args:
            title: Result title
            content: Result content
            query_words: Set of query words
            base_score: Base score from search engine

        Returns:
            Calculated relevance score
        """
        try:
            # Start with base score
            relevance = base_score

            # Title relevance (higher weight)
            title_words = set(title.lower().split())
            title_overlap = len(query_words.intersection(title_words))
            title_boost = (title_overlap / max(len(query_words), 1)) * 0.3

            # Content relevance
            content_words = set(content.lower().split())
            content_overlap = len(query_words.intersection(content_words))
            content_boost = (content_overlap / max(len(query_words), 1)) * 0.2

            # Exact phrase matching bonus
            query_phrase = " ".join(query_words).lower()
            if query_phrase in title.lower():
                relevance += 0.2
            elif query_phrase in content.lower():
                relevance += 0.1

            # Final score calculation
            final_score = min(relevance + title_boost + content_boost, 1.0)

            return round(final_score, 3)

        except Exception as e:
            self.logger.warning(f"Error calculating relevance score: {str(e)}")
            return base_score

    def get_search_context(self, query: str, **kwargs) -> str:
        """
        Get search context suitable for RAG applications.

        Args:
            query: Search query string
            **kwargs: Additional search parameters

        Returns:
            Formatted context string
        """
        search_results = self.search(query, **kwargs)

        if not search_results.get("results"):
            error_msg = search_results.get("error", "Unknown error")
            return f"No live search results found for: {query}. Error: {error_msg}"

        context_parts = []

        # Add answer if available
        if search_results.get("answer"):
            context_parts.append(f"Answer: {search_results['answer']}")
            context_parts.append("")

        # Add search results
        context_parts.append("Search Results:")
        for i, result in enumerate(search_results["results"], 1):
            context_parts.append(f"{i}. {result['title']}")
            context_parts.append(f"   URL: {result['url']}")
            context_parts.append(f"   Content: {result['content']}")
            if result.get("published_date"):
                context_parts.append(f"   Published: {result['published_date']}")
            context_parts.append("")

        # Add metadata
        metadata = search_results.get("search_metadata", {})
        context_parts.append(
            f"Search performed at: {metadata.get('timestamp', 'Unknown')}"
        )
        context_parts.append(f"Source: {metadata.get('source', 'Unknown')}")
        context_parts.append(f"Results count: {metadata.get('results_count', 0)}")

        return "\n".join(context_parts)

    def qna_search(self, query: str, **kwargs) -> str:
        """
        Get a quick answer to a question using Tavily's QnA search.

        Args:
            query: Question to answer
            **kwargs: Additional search parameters

        Returns:
            Answer string
        """
        if not self.is_enabled():
            return "Live search is disabled or not properly configured."

        try:
            # Use Tavily's QnA search method
            answer = self.tavily_client.qna_search(query=query)
            return answer if answer else "No answer found for the given question."

        except Exception as e:
            self.logger.error(f"Error in QnA search: {str(e)}")
            return f"Error getting answer: {str(e)}"

    def _generate_cache_key(
        self, query: str, max_results: int, search_depth: str
    ) -> str:
        """Generate cache key for search results."""
        import hashlib

        cache_string = f"{query.lower().strip()}{max_results}{search_depth}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached result is still valid (30 minutes for live search)."""
        return datetime.now() - timestamp < timedelta(minutes=30)

    def _add_to_history(self, query: str, result_count: int):
        """Add search to history for analytics."""
        self.search_history.append(
            {
                "query": query,
                "timestamp": datetime.now(),
                "result_count": result_count,
                "search_type": "live_web",
            }
        )

        # Keep only last 50 searches
        if len(self.search_history) > 50:
            self.search_history = self.search_history[-50:]

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the live search service.

        Returns:
            Dictionary containing health status
        """
        try:
            if not self.enabled:
                return {
                    "status": "disabled",
                    "message": "Live search is disabled in configuration",
                    "timestamp": datetime.now().isoformat(),
                }

            if not self.tavily_client:
                return {
                    "status": "error",
                    "message": "Tavily client not initialized. Check TAVILY_API_KEY environment variable.",
                    "timestamp": datetime.now().isoformat(),
                }

            # Perform a simple test search
            test_result = self.search("test health check", max_results=1)

            if test_result.get("error"):
                return {
                    "status": "error",
                    "message": f"Health check failed: {test_result['error']}",
                    "timestamp": datetime.now().isoformat(),
                }

            return {
                "status": "healthy",
                "message": "Live search service is operational",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "max_results": self.max_results,
                    "search_depth": self.search_depth,
                    "include_answer": self.include_answer,
                    "topic": self.topic,
                },
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            }

    def get_search_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about search patterns.

        Returns:
            Dictionary with search analytics
        """
        if not self.search_history:
            return {"total_searches": 0, "cache_hit_rate": 0.0, "average_results": 0.0}

        total_searches = len(self.search_history)
        avg_results = (
            sum(s["result_count"] for s in self.search_history) / total_searches
        )

        # Recent search trends
        recent_searches = [s["query"] for s in self.search_history[-10:]]

        return {
            "total_searches": total_searches,
            "average_results_per_search": round(avg_results, 2),
            "recent_searches": recent_searches,
            "cache_size": len(self.search_cache),
            "search_type": "live_web",
        }

    def clear_cache(self):
        """Clear the search cache."""
        self.search_cache.clear()
        self.logger.info("Live search cache cleared")

    def clear_history(self):
        """Clear the search history."""
        self.search_history.clear()
        self.logger.info("Live search history cleared")


# 🔄 Compatibility alias for existing imports
LiveSearchManager = LiveSearchProcessor
