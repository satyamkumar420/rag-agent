"""
MCP Tavily Integration Module

This module demonstrates how to integrate Tavily API via MCP (Model Context Protocol)
for live web search functionality in the RAG system.

Technology: MCP + Tavily API
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime


class MCPTavilyIntegration:
    """
    Handles MCP integration with Tavily API for live web search.

    This class provides the bridge between the RAG system and Tavily's
    search capabilities through the Model Context Protocol.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MCP Tavily integration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 🔧 MCP Configuration
        self.server_name = self.config.get("mcp_server_name", "tavily-mcp")
        self.tool_name = self.config.get("mcp_tool_name", "tavily-search")
        self.timeout = self.config.get("timeout", 30)

        self.logger.info(" MCP Tavily Integration initialized")

    def search_web(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        time_range: str = "month",
        topic: str = "general",
    ) -> Dict[str, Any]:
        """
        Perform web search using Tavily API via MCP.

        Args:
            query: Search query
            max_results: Maximum number of results
            search_depth: Search depth (basic/advanced)
            time_range: Time range for results
            topic: Search topic category

        Returns:
            Dictionary with search results
        """
        try:
            self.logger.info(f" MCP Tavily search: '{query}' (depth: {search_depth})")

            # 🚀 Prepare MCP arguments
            mcp_arguments = {
                "query": query,
                "max_results": min(max_results, 20),  # Tavily limit
                "search_depth": search_depth,
                "topic": topic,
                "include_raw_content": True,
                "time_range": time_range,
            }

            # 🌐 This is where the actual MCP call would be made
            # In a real implementation, this would use the MCP client:

            """
            Example MCP call structure:
            
            result = use_mcp_tool(
                server_name=self.server_name,
                tool_name=self.tool_name,
                arguments=mcp_arguments
            )
            """

            # 🚧 For demonstration, we'll simulate the MCP response structure
            simulated_result = self._simulate_tavily_response(query, max_results)

            # 🔄 Process and validate MCP response
            processed_result = self._process_mcp_response(simulated_result, query)

            self.logger.info(
                f" MCP search completed: {processed_result.get('total_results', 0)} results"
            )
            return processed_result

        except Exception as e:
            self.logger.error(f" MCP Tavily search failed: {str(e)}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": str(e),
                "status": "mcp_error",
            }

    def _simulate_tavily_response(self, query: str, max_results: int) -> Dict[str, Any]:
        """
        Simulate Tavily API response for demonstration.

        In production, this would be replaced by actual MCP call results.
        """
        # 🚧 Simulated response structure matching Tavily API
        return {
            "query": query,
            "follow_up_questions": None,
            "answer": f"Based on web search for '{query}'...",
            "images": [],
            "results": [
                {
                    "title": f"Example Result 1 for {query}",
                    "url": "https://example.com/result1",
                    "content": f"This is example content related to {query}. It provides comprehensive information about the topic.",
                    "raw_content": f"Raw content for {query} with additional details...",
                    "published_date": "2024-01-15",
                    "score": 0.95,
                },
                {
                    "title": f"Example Result 2 for {query}",
                    "url": "https://example.com/result2",
                    "content": f"Another relevant result for {query} with different perspective and insights.",
                    "raw_content": f"Extended raw content for {query}...",
                    "published_date": "2024-01-14",
                    "score": 0.87,
                },
            ][:max_results],
            "response_time": 1.2,
        }

    def _process_mcp_response(
        self, mcp_result: Dict[str, Any], original_query: str
    ) -> Dict[str, Any]:
        """
        Process and validate MCP response from Tavily.

        Args:
            mcp_result: Raw MCP response
            original_query: Original search query

        Returns:
            Processed search results
        """
        try:
            # 🔍 Extract results from MCP response
            raw_results = mcp_result.get("results", [])

            # 🔄 Process each result
            processed_results = []
            for i, result in enumerate(raw_results):
                processed_result = {
                    "title": result.get("title", f"Web Result {i+1}"),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "raw_content": result.get("raw_content", ""),
                    "score": result.get("score", 0.0),
                    "published_date": result.get("published_date", ""),
                    "rank": i + 1,
                    "source": "tavily_web_search",
                    "search_engine": "tavily",
                    "metadata": {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content_length": len(result.get("content", "")),
                        "has_raw_content": bool(result.get("raw_content")),
                        "search_rank": i + 1,
                        "published_date": result.get("published_date", ""),
                    },
                }
                processed_results.append(processed_result)

            # 📊 Prepare final response
            return {
                "query": original_query,
                "results": processed_results,
                "total_results": len(processed_results),
                "answer": mcp_result.get("answer", ""),
                "follow_up_questions": mcp_result.get("follow_up_questions", []),
                "response_time": mcp_result.get("response_time", 0),
                "timestamp": datetime.now(),
                "status": "success",
                "source": "mcp_tavily",
            }

        except Exception as e:
            self.logger.error(f" Error processing MCP response: {str(e)}")
            return {
                "query": original_query,
                "results": [],
                "total_results": 0,
                "error": f"Response processing failed: {str(e)}",
                "status": "processing_error",
            }

    def test_connection(self) -> Dict[str, Any]:
        """
        Test MCP connection to Tavily.

        Returns:
            Connection test results
        """
        try:
            self.logger.info(" Testing MCP Tavily connection...")

            # 🔍 Simple test query
            test_result = self.search_web(
                query="test connection", max_results=1, search_depth="basic"
            )

            if test_result.get("status") == "success":
                return {
                    "status": "success",
                    "message": " MCP Tavily connection successful",
                    "server_name": self.server_name,
                    "tool_name": self.tool_name,
                    "response_time": test_result.get("response_time", 0),
                }
            else:
                return {
                    "status": "error",
                    "message": " MCP Tavily connection failed",
                    "error": test_result.get("error", "Unknown error"),
                }

        except Exception as e:
            self.logger.error(f" MCP connection test failed: {str(e)}")
            return {
                "status": "error",
                "message": " MCP connection test failed",
                "error": str(e),
            }

    def get_server_info(self) -> Dict[str, Any]:
        """
        Get MCP server information.

        Returns:
            Server information dictionary
        """
        return {
            "server_name": self.server_name,
            "tool_name": self.tool_name,
            "timeout": self.timeout,
            "status": "configured",
            "description": "MCP integration for Tavily web search API",
        }


# 🔧 Helper function for easy integration
def create_mcp_tavily_client(
    config: Optional[Dict[str, Any]] = None,
) -> MCPTavilyIntegration:
    """
    Create and configure MCP Tavily client.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured MCPTavilyIntegration instance
    """
    return MCPTavilyIntegration(config)


# 📝 Example usage and integration guide
if __name__ == "__main__":
    """
    Example usage of MCP Tavily Integration

    This demonstrates how to use the MCP integration in your RAG system.
    """

    # 🔧 Configure MCP client
    config = {
        "mcp_server_name": "tavily-mcp",
        "mcp_tool_name": "tavily-search",
        "timeout": 30,
    }

    # 🚀 Create client
    mcp_client = create_mcp_tavily_client(config)

    # 🧪 Test connection
    connection_test = mcp_client.test_connection()
    print(f"Connection test: {connection_test}")

    # 🔍 Example search
    search_result = mcp_client.search_web(
        query="latest AI developments 2024",
        max_results=5,
        search_depth="basic",
        time_range="month",
    )

    print(f"Search results: {search_result.get('total_results', 0)} found")
    for result in search_result.get("results", []):
        print(f"- {result['title']}: {result['url']}")
