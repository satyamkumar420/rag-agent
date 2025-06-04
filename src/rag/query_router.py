"""
Query Router Module

This module intelligently routes queries between local document search
and live web search based on query analysis and user preferences.

Technology: Custom routing logic with RAG + Live Search integration
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum


class QueryType(Enum):
    """Enumeration of different query types for routing decisions."""

    FACTUAL = "factual"  # 📊 Current facts, news, data
    CONCEPTUAL = "conceptual"  # 💡 Definitions, explanations
    PROCEDURAL = "procedural"  # 🔧 How-to, instructions
    ANALYTICAL = "analytical"  # 📈 Analysis, comparisons
    TEMPORAL = "temporal"  # ⏰ Time-sensitive information
    HYBRID = "hybrid"  # 🔄 Requires both sources


class QueryRouter:
    """
    Intelligent query router that decides between local docs and live search.

    Features:
    - Query type classification
    - Intelligent routing decisions
    - Hybrid search coordination
    - Result fusion and ranking
    - Performance optimization
    """

    def __init__(
        self,
        local_query_processor,
        live_search_processor,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the QueryRouter.

        Args:
            local_query_processor: Local document query processor
            live_search_processor: Live web search processor
            config: Configuration dictionary
        """
        self.local_processor = local_query_processor
        self.live_processor = live_search_processor
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 🎯 Routing configuration
        self.enable_hybrid_search = self.config.get("enable_hybrid_search", True)
        self.local_weight = self.config.get("local_weight", 0.6)
        self.live_weight = self.config.get("live_weight", 0.4)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.max_hybrid_results = self.config.get("max_hybrid_results", 10)

        # 📊 Analytics and caching
        self.routing_history = []
        self.routing_cache = {}

        # 🔍 Query classification patterns
        self._init_classification_patterns()

        self.logger.info("QueryRouter initialized with intelligent routing")

    def _init_classification_patterns(self):
        """Initialize patterns for query classification."""
        self.temporal_keywords = {
            "current",
            "latest",
            "recent",
            "today",
            "now",
            "2025",
            "breaking",
            "news",
            "update",
            "trending",
            "happening",
        }

        self.factual_keywords = {
            "what is",
            "who is",
            "when did",
            "where is",
            "statistics",
            "data",
            "facts",
            "numbers",
            "rate",
            "percentage",
        }

        self.procedural_keywords = {
            "how to",
            "steps",
            "guide",
            "tutorial",
            "instructions",
            "process",
            "method",
            "way to",
            "procedure",
        }

        self.conceptual_keywords = {
            "explain",
            "definition",
            "meaning",
            "concept",
            "theory",
            "principle",
            "idea",
            "understand",
            "clarify",
        }

    def route_query(
        self,
        query: str,
        use_live_search: bool = False,
        max_results: int = 5,
        search_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Route query to appropriate search method(s).

        Args:
            query: User query string
            use_live_search: Force live search usage
            max_results: Maximum results to return
            search_options: Additional search options

        Returns:
            Dictionary with routed results and metadata
        """
        if not query or not query.strip():
            return {
                "query": query,
                "results": [],
                "routing_decision": "error",
                "error": "Empty query provided",
            }

        self.logger.info(f" Routing query: {query[:100]}...")
        start_time = time.time()

        try:
            # 🎯 Classify query type
            query_type = self._classify_query(query)

            # 🔄 Make routing decision
            routing_decision = self._make_routing_decision(
                query, query_type, use_live_search
            )

            # 🚀 Execute search based on routing decision
            if routing_decision == "local_only":
                result = self._search_local_only(query, max_results)
            elif routing_decision == "live_only":
                result = self._search_live_only(query, max_results, search_options)
            elif routing_decision == "hybrid":
                result = self._search_hybrid(query, max_results, search_options)
            else:
                result = self._search_fallback(query, max_results)

            # 📊 Add routing metadata
            result.update(
                {
                    "query_type": query_type.value,
                    "routing_decision": routing_decision,
                    "processing_time": time.time() - start_time,
                    "timestamp": datetime.now(),
                }
            )

            # 📈 Track routing decision
            self._track_routing_decision(query, query_type, routing_decision)

            self.logger.info(
                f" Query routed via {routing_decision} in {result['processing_time']:.2f}s"
            )
            return result

        except Exception as e:
            self.logger.error(f" Error in query routing: {str(e)}")
            return {
                "query": query,
                "results": [],
                "routing_decision": "error",
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def _classify_query(self, query: str) -> QueryType:
        """
        Classify query type for routing decisions.

        Args:
            query: Query string to classify

        Returns:
            QueryType enum value
        """
        query_lower = query.lower()

        # 🔍 Check for temporal indicators
        if any(keyword in query_lower for keyword in self.temporal_keywords):
            return QueryType.TEMPORAL

        # 📊 Check for factual queries
        if any(keyword in query_lower for keyword in self.factual_keywords):
            return QueryType.FACTUAL

        # 🔧 Check for procedural queries
        if any(keyword in query_lower for keyword in self.procedural_keywords):
            return QueryType.PROCEDURAL

        # 💡 Check for conceptual queries
        if any(keyword in query_lower for keyword in self.conceptual_keywords):
            return QueryType.CONCEPTUAL

        # 📈 Default to analytical for complex queries
        if len(query.split()) > 10:
            return QueryType.ANALYTICAL

        # 🔄 Default to hybrid for uncertain cases
        return QueryType.HYBRID

    def _make_routing_decision(
        self, query: str, query_type: QueryType, force_live: bool
    ) -> str:
        """
        Make intelligent routing decision based on query analysis.

        Args:
            query: Query string
            query_type: Classified query type
            force_live: Whether to force live search

        Returns:
            Routing decision string
        """
        # 🚀 Force live search if requested
        if force_live:
            return "live_only"

        # 🎯 Route based on query type
        if query_type == QueryType.TEMPORAL:
            return "live_only"  # ⏰ Time-sensitive info needs live search

        elif query_type == QueryType.FACTUAL:
            return "hybrid"  # 📊 Facts benefit from both sources

        elif query_type == QueryType.PROCEDURAL:
            return "local_only"  # 🔧 Procedures likely in documents

        elif query_type == QueryType.CONCEPTUAL:
            return "local_only"  # 💡 Concepts likely in documents

        elif query_type == QueryType.ANALYTICAL:
            return "hybrid"  # 📈 Analysis benefits from both

        else:  # QueryType.HYBRID
            return "hybrid"  # 🔄 Default to hybrid

    def _search_local_only(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search only local documents."""
        self.logger.info(" Searching local documents only")

        try:
            local_result = self.local_processor.process_query(query)

            # 🔄 Format results consistently
            formatted_results = []
            for item in local_result.get("context", [])[:max_results]:
                formatted_results.append(
                    {
                        "title": f"Document: {item.get('source', 'Unknown')}",
                        "content": item.get("text", ""),
                        "score": item.get("score", 0.0),
                        "source": item.get("source", "local_document"),
                        "type": "local_document",
                        "metadata": item.get("metadata", {}),
                    }
                )

            return {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "sources": ["local_documents"],
                "local_results": local_result.get("total_results", 0),
            }

        except Exception as e:
            self.logger.error(f" Local search error: {str(e)}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": f"Local search failed: {str(e)}",
            }

    def _search_live_only(
        self, query: str, max_results: int, search_options: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Search only live web sources."""
        self.logger.info(" Searching live web sources only")

        try:
            # 🎯 Extract search options
            options = search_options or {}
            search_depth = options.get("search_depth", "basic")
            time_range = options.get("time_range", "month")

            live_result = self.live_processor.search_web(
                query,
                max_results=max_results,
                search_depth=search_depth,
                time_range=time_range,
            )

            return {
                "query": query,
                "results": live_result.get("results", []),
                "total_results": live_result.get("total_results", 0),
                "sources": ["live_web"],
                "live_results": live_result.get("total_results", 0),
                "search_params": live_result.get("search_params", {}),
            }

        except Exception as e:
            self.logger.error(f" Live search error: {str(e)}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": f"Live search failed: {str(e)}",
            }

    def _search_hybrid(
        self, query: str, max_results: int, search_options: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform hybrid search combining local and live sources."""
        self.logger.info(" Performing hybrid search")

        try:
            # 📊 Calculate result distribution
            local_count = int(max_results * self.local_weight)
            live_count = max_results - local_count

            # 🚀 Perform both searches concurrently (simplified sequential for now)
            local_result = self.local_processor.process_query(query)

            options = search_options or {}
            live_result = self.live_processor.search_web(
                query,
                max_results=live_count,
                search_depth=options.get("search_depth", "basic"),
                time_range=options.get("time_range", "month"),
            )

            # 🔄 Combine and rank results
            combined_results = self._fuse_results(
                local_result, live_result, local_count, live_count
            )

            return {
                "query": query,
                "results": combined_results[:max_results],
                "total_results": len(combined_results),
                "sources": ["local_documents", "live_web"],
                "local_results": local_result.get("total_results", 0),
                "live_results": live_result.get("total_results", 0),
                "fusion_method": "weighted_ranking",
            }

        except Exception as e:
            self.logger.error(f" Hybrid search error: {str(e)}")
            return self._search_fallback(query, max_results)

    def _fuse_results(
        self,
        local_result: Dict[str, Any],
        live_result: Dict[str, Any],
        local_count: int,
        live_count: int,
    ) -> List[Dict[str, Any]]:
        """
        Fuse results from local and live searches.

        Args:
            local_result: Results from local search
            live_result: Results from live search
            local_count: Number of local results to include
            live_count: Number of live results to include

        Returns:
            Fused and ranked results
        """
        fused_results = []

        # 📚 Process local results
        for item in local_result.get("context", [])[:local_count]:
            fused_results.append(
                {
                    "title": f"Document: {item.get('source', 'Unknown')}",
                    "content": item.get("text", ""),
                    "score": item.get("score", 0.0) * self.local_weight,
                    "source": item.get("source", "local_document"),
                    "type": "local_document",
                    "metadata": item.get("metadata", {}),
                    "fusion_score": item.get("score", 0.0) * self.local_weight,
                }
            )

        # 🌐 Process live results
        for item in live_result.get("results", [])[:live_count]:
            fused_results.append(
                {
                    "title": item.get("title", "Web Result"),
                    "content": item.get("content", ""),
                    "score": item.get("relevance_score", 0.0) * self.live_weight,
                    "source": item.get("url", "web_search"),
                    "type": "web_result",
                    "metadata": item.get("metadata", {}),
                    "fusion_score": item.get("relevance_score", 0.0) * self.live_weight,
                }
            )

        # 🔄 Sort by fusion score
        fused_results.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)

        return fused_results

    def _search_fallback(self, query: str, max_results: int) -> Dict[str, Any]:
        """Fallback search method when other methods fail."""
        self.logger.warning(" Using fallback search method")

        try:
            # 📚 Try local search first
            local_result = self.local_processor.process_query(query)

            if local_result.get("context"):
                return self._search_local_only(query, max_results)
            else:
                return {
                    "query": query,
                    "results": [],
                    "total_results": 0,
                    "sources": [],
                    "error": "No results found in fallback search",
                }

        except Exception as e:
            self.logger.error(f" Fallback search failed: {str(e)}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": f"All search methods failed: {str(e)}",
            }

    def _track_routing_decision(
        self, query: str, query_type: QueryType, routing_decision: str
    ):
        """Track routing decisions for analytics."""
        self.routing_history.append(
            {
                "query": query[:100],  # Truncate for privacy
                "query_type": query_type.value,
                "routing_decision": routing_decision,
                "timestamp": datetime.now(),
            }
        )

        # 📊 Keep only last 100 routing decisions
        if len(self.routing_history) > 100:
            self.routing_history = self.routing_history[-100:]

    def get_routing_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about routing patterns.

        Returns:
            Dictionary with routing analytics
        """
        if not self.routing_history:
            return {
                "total_queries": 0,
                "routing_distribution": {},
                "query_type_distribution": {},
            }

        total_queries = len(self.routing_history)

        # 📊 Calculate routing distribution
        routing_counts = {}
        query_type_counts = {}

        for entry in self.routing_history:
            routing = entry["routing_decision"]
            query_type = entry["query_type"]

            routing_counts[routing] = routing_counts.get(routing, 0) + 1
            query_type_counts[query_type] = query_type_counts.get(query_type, 0) + 1

        # 📈 Convert to percentages
        routing_distribution = {
            k: round((v / total_queries) * 100, 1) for k, v in routing_counts.items()
        }

        query_type_distribution = {
            k: round((v / total_queries) * 100, 1) for k, v in query_type_counts.items()
        }

        return {
            "total_queries": total_queries,
            "routing_distribution": routing_distribution,
            "query_type_distribution": query_type_distribution,
            "recent_decisions": [
                {
                    "query": entry["query"][:50] + "...",
                    "type": entry["query_type"],
                    "routing": entry["routing_decision"],
                }
                for entry in self.routing_history[-5:]
            ],
        }

    def clear_cache(self):
        """Clear routing cache."""
        self.routing_cache.clear()
        self.logger.info(" Routing cache cleared")

    def clear_history(self):
        """Clear routing history."""
        self.routing_history.clear()
        self.logger.info(" Routing history cleared")
