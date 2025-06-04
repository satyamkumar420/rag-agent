# 🌐 Live Search Implementation Guide

This document explains the implementation of live web search functionality using Tavily API integration in the RAG AI system.

## 🚨 **Architecture Overview**

The live search feature follows the architecture diagram you provided:

```
           +------------------+
           | User's Query 🔍  |
           +--------+---------+
                    |
        +-----------v-----------+
        |      Query Router     |
        +-----------+-----------+
                    |
        +-----------+-----------+
        |     Docs VectorDB     | (Gemini + Pinecone) 📄
        +-----------+-----------+
                    |
        +-----------+-----------+
        |     Live Web Search   | (Tavily) 🌐
        +-----------+-----------+
                    |
        +-----------v-----------+
        |   RAG Fusion Answer   | 🧠✨
        +-----------+-----------+
                    |
            +-------v-------+
            |   Gradio UI   |
            +---------------+
```

## 🧭 **Implementation Components**

### 1. **Query Router** (`src/rag/query_router.py`)

- 🎯 **Purpose**: Intelligently routes queries between local docs and live search
- 🔧 **Features**:
  - Query type classification (temporal, factual, procedural, etc.)
  - Intelligent routing decisions
  - Hybrid search coordination
  - Result fusion and ranking

### 2. **Live Search Processor** (`src/rag/live_search.py`)

- 🌐 **Purpose**: Handles live web search using Tavily API
- 🔧 **Features**:
  - Real-time web search with Tavily API
  - Search result filtering and ranking
  - Content extraction and summarization
  - Search result caching for performance

### 3. **MCP Tavily Integration** (`src/integrations/mcp_tavily_integration.py`)

- 🔗 **Purpose**: Bridge between RAG system and Tavily API via MCP
- 🔧 **Features**:
  - MCP protocol integration
  - Tavily API communication
  - Response processing and validation
  - Connection testing and monitoring

### 4. **Enhanced UI** (`src/ui/gradio_app.py`)

- 🎨 **Purpose**: User interface with live search options
- 🔧 **Features**:
  - Live search checkbox in query options
  - Search depth and time range controls
  - Enhanced result display with search type indicators
  - Real-time search status updates

## 🛠️ **Implementation Details**

### **Query Processing Flow**

1. **User Input** 🔍

   - User enters query in Gradio interface
   - Selects live search option (checkbox)
   - Configures search depth and time range

2. **Query Routing** 🧭

   - Query Router analyzes query type
   - Makes intelligent routing decision:
     - `local_only`: Search only local documents
     - `live_only`: Search only web sources
     - `hybrid`: Combine both sources

3. **Search Execution** 🚀

   - **Local Search**: Uses existing RAG pipeline
   - **Live Search**: Calls Tavily API via MCP
   - **Hybrid**: Executes both and fuses results

4. **Response Generation** 🧠
   - Processes search results
   - Generates coherent response
   - Includes source attribution

### **MCP Integration**

The system uses MCP (Model Context Protocol) to integrate with Tavily:

```python
# Example MCP call structure
use_mcp_tool(
    server_name="tavily-mcp",
    tool_name="tavily-search",
    arguments={
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "time_range": time_range,
        "topic": "general"
    }
)
```

### **Configuration**

Live search settings in `config/config.yaml`:

```yaml
live_search:
  enable_caching: true
  include_raw_content: true
  max_results: 10
  search_depth: basic
  time_range: month

query_router:
  confidence_threshold: 0.5
  enable_hybrid_search: true
  live_weight: 0.4
  local_weight: 0.6
  max_hybrid_results: 10
```

## 🚀 **Usage Guide**

### **For Users**

1. **Enable Live Search**:

   - Go to "❓ Ask Questions" tab
   - Expand "⚙️ Query Options"
   - Check "🔍 Enable Live Web Search"

2. **Configure Search Options**:

   - **Search Depth**: Basic (faster) or Advanced (comprehensive)
   - **Time Range**: Day, Week, Month, or Year

3. **Ask Questions**:
   - Enter your query
   - Click "🚀 Get Answer"
   - View results with live web data

### **For Developers**

1. **Setup MCP Server**:

   ```bash
   npx -y tavily-mcp
   ```

2. **Configure API Keys**:

   ```bash
   export TAVILY_API_KEY="your_tavily_api_key"
   ```

3. **Test Integration**:

   ```python
   from src.integrations import create_mcp_tavily_client

   client = create_mcp_tavily_client()
   result = client.test_connection()
   print(result)
   ```

## 🔧 **Technical Implementation**

### **Key Classes and Methods**

#### **QueryRouter**

```python
def route_query(
    self,
    query: str,
    use_live_search: bool = False,
    max_results: int = 5,
    search_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

#### **LiveSearchProcessor**

```python
def search_web(
    self,
    query: str,
    max_results: Optional[int] = None,
    search_depth: Optional[str] = None,
    time_range: Optional[str] = None
) -> Dict[str, Any]
```

#### **MCPTavilyIntegration**

```python
def search_web(
    self,
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    time_range: str = "month",
    topic: str = "general"
) -> Dict[str, Any]
```

### **Query Classification**

The system automatically classifies queries:

- **🕐 Temporal**: Current events, recent news
- **📊 Factual**: Statistics, data, facts
- **🔧 Procedural**: How-to, instructions
- **💡 Conceptual**: Definitions, explanations
- **📈 Analytical**: Analysis, comparisons
- **🔄 Hybrid**: Requires multiple sources

### **Result Fusion**

For hybrid searches, results are fused using weighted ranking:

```python
# Local results weight: 60%
# Live results weight: 40%
fusion_score = (local_score * 0.6) + (live_score * 0.4)
```

## 📊 **Performance Considerations**

### **Caching Strategy**

- **Local Cache**: 30-minute TTL for live search results
- **Query Cache**: Stores processed queries to avoid re-processing
- **Result Cache**: Caches formatted results for quick retrieval

### **Rate Limiting**

- Respects Tavily API rate limits
- Implements exponential backoff for failed requests
- Queues requests during high load

### **Fallback Mechanisms**

- Falls back to local search if live search fails
- Graceful degradation when MCP server is unavailable
- Error handling with user-friendly messages

## 🧪 **Testing**

### **Unit Tests**

```bash
python -m pytest tests/test_live_search.py
python -m pytest tests/test_query_router.py
python -m pytest tests/test_mcp_integration.py
```

### **Integration Tests**

```bash
python -m pytest tests/integration/test_live_search_flow.py
```

### **Manual Testing**

1. Test live search checkbox functionality
2. Verify search depth and time range controls
3. Test hybrid search result fusion
4. Validate error handling and fallbacks

## 🔒 **Security Considerations**

### **API Key Management**

- Store Tavily API key securely in environment variables
- Never log or expose API keys in responses
- Implement key rotation procedures

### **Input Validation**

- Sanitize user queries before sending to external APIs
- Validate search parameters and limits
- Prevent injection attacks

### **Rate Limiting**

- Implement client-side rate limiting
- Monitor API usage and costs
- Set reasonable default limits

## 📈 **Monitoring and Analytics**

### **Metrics Tracked**

- Live search usage frequency
- Query routing decisions
- Response times and success rates
- User satisfaction with live results

### **Logging**

- Query routing decisions
- Live search API calls and responses
- Error rates and failure modes
- Performance metrics

## 🚀 **Future Enhancements**

### **Planned Features**

1. **🔍 Advanced Search Filters**: Domain filtering, content type selection
2. **📊 Search Analytics**: Usage patterns, popular queries
3. **🤖 Smart Caching**: ML-based cache optimization
4. **🌍 Multi-Language Support**: International search capabilities
5. **📱 Mobile Optimization**: Responsive design improvements

### **Integration Opportunities**

- **🔗 Additional Search Engines**: Google, Bing, DuckDuckGo
- **📰 News APIs**: Real-time news integration
- **🏢 Enterprise Search**: Internal knowledge bases
- **📚 Academic Sources**: Research paper integration

## 🆘 **Troubleshooting**

### **Common Issues**

1. **Live Search Not Working**:

   - Check Tavily API key configuration
   - Verify MCP server is running
   - Test network connectivity

2. **Slow Response Times**:

   - Reduce search depth to "basic"
   - Lower max_results limit
   - Check cache configuration

3. **Poor Result Quality**:
   - Adjust query routing weights
   - Refine search parameters
   - Update query classification patterns

### **Debug Commands**

```bash
# Test MCP connection
python -c "from src.integrations import create_mcp_tavily_client; print(create_mcp_tavily_client().test_connection())"

# Check configuration
python -c "from src.utils.config_manager import ConfigManager; print(ConfigManager().get_section('live_search'))"

# Test query routing
python -c "from src.rag.query_router import QueryRouter; print('Router loaded successfully')"
```

## 📚 **Additional Resources**

- **Tavily API Documentation**: [https://docs.tavily.com](https://docs.tavily.com)
- **MCP Protocol Specification**: [https://modelcontextprotocol.io](https://modelcontextprotocol.io)
- **RAG System Architecture**: See `docs/architecture.md`
- **Configuration Guide**: See `SETTINGS_MANAGEMENT.md`

---

**🎯 Implementation Status**: ✅ **COMPLETE**

The live search feature is now fully implemented and ready for use! Users can enable live web search through the checkbox in the query options, and the system will intelligently route queries between local documents and live web sources using Tavily API.
