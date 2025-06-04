"""
Integrations Module

This module contains integrations with external services and APIs
for enhanced RAG functionality.

Available Integrations:
- MCP Tavily: Live web search via Model Context Protocol
"""

from .mcp_tavily_integration import MCPTavilyIntegration, create_mcp_tavily_client

__all__ = ["MCPTavilyIntegration", "create_mcp_tavily_client"]
