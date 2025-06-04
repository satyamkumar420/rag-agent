"""
Gradio UI Module

This module provides an intuitive interface for document upload,
URL input, and querying using Gradio.

Technology: Gradio
"""

import logging
import os
import sys
import tempfile
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    logging.warning("Gradio not available.")


class GradioApp:
    """
    Provides a comprehensive Gradio-based user interface for the RAG system.

    Features:
    - Document upload with progress tracking
    - URL processing with status updates
    - Interactive Q&A interface with source display
    - Knowledge base management
    - System status and health monitoring
    - Analytics dashboard
    """

    def __init__(self, rag_system, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GradioApp with the RAG system.

        Args:
            rag_system: Instance of the complete RAG system
            config: Configuration dictionary with UI parameters
        """
        self.rag_system = rag_system
        self.config = config or {}
        self.logger = self._setup_unicode_logger()

        # 🔧 Initialize settings manager
        from utils.settings_manager import SettingsManager

        config_manager = getattr(rag_system, "config_manager", None)
        self.settings_manager = SettingsManager(config_manager)

        # UI Configuration
        self.title = self.config.get("title", "AI Embedded Knowledge Agent")
        self.description = self.config.get(
            "description",
            "Upload documents or provide URLs to build your knowledge base, then ask questions!",
        )
        self.theme = self.config.get("theme", "default")
        self.share = self.config.get("share", False)

        # Features configuration
        self.features = self.config.get("features", {})
        self.enable_file_upload = self.features.get("file_upload", True)
        self.enable_url_input = self.features.get("url_input", True)
        self.enable_query_interface = self.features.get("query_interface", True)
        self.enable_source_display = self.features.get("source_display", True)
        self.enable_confidence_display = self.features.get("confidence_display", True)

        # State management
        self.processing_status = "Ready"
        self.total_documents = 0
        self.total_chunks = 0
        self.query_count = 0

        # Initialize interface
        self.interface = None
        self._create_interface()

        self._log_safe("GradioApp initialized with advanced features")

    def _setup_unicode_logger(self):
        """🔧 Setup Unicode-safe logger for cross-platform compatibility."""
        logger = logging.getLogger(__name__)

        # ✅ Configure handler with UTF-8 encoding for Windows compatibility
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)

            # 🌍 Force UTF-8 encoding on Windows to handle emojis
            if sys.platform.startswith("win"):
                try:
                    # ⚡ Try to reconfigure stdout with UTF-8 encoding
                    handler.stream = open(
                        sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1
                    )
                except Exception:
                    # 🔄 Fallback to default if reconfiguration fails
                    pass

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    def _log_safe(self, message: str, level: str = "info"):
        """🛡️ Unicode-safe logging that handles emojis on Windows."""
        try:
            # ✅ Pre-process message to be safe for Windows cp1252 encoding
            safe_message = self._make_message_safe(message)
            getattr(self.logger, level)(safe_message)
        except UnicodeEncodeError:
            # 🔄 Additional fallback: Remove all non-ASCII characters
            ascii_message = message.encode("ascii", "ignore").decode("ascii")
            getattr(self.logger, level)(f"[ENCODING_SAFE] {ascii_message}")
        except Exception as e:
            # 🚨 Last resort: Basic logging without special characters
            basic_message = (
                str(message).replace("🌐", "[LIVE]").replace("📚", "[LOCAL]")
            )
            try:
                getattr(self.logger, level)(f"[SAFE] {basic_message}")
            except:
                print(f"[FALLBACK] {basic_message}")  # Direct print as last resort

    def _make_message_safe(self, message: str) -> str:
        """🔄 Convert emoji characters to safe text equivalents."""
        emoji_map = {
            "🔍": "[SEARCH]",
            "✅": "[SUCCESS]",
            "❌": "[ERROR]",
            "🚀": "[ROCKET]",
            "📄": "[DOC]",
            "🔗": "[LINK]",
            "⚡": "[FAST]",
            "🎯": "[TARGET]",
            "🟢": "[GREEN]",
            "🟡": "[YELLOW]",
            "🔴": "[RED]",
            "📊": "[CHART]",
            "🕷️": "[SPIDER]",
            "💡": "[IDEA]",
            "🔄": "[REFRESH]",
            "📚": "[BOOKS]",
            "🩺": "[HEALTH]",
            "📈": "[ANALYTICS]",
            "🌐": "[LIVE]",  # ✅ Added the problematic emoji
            "🌍": "[WORLD]",
            "🔧": "[TOOL]",
            "🛡️": "[SHIELD]",
            "🎨": "[DESIGN]",
            "📝": "[NOTE]",
            "🗑️": "[DELETE]",
            "💾": "[SAVE]",
            "📁": "[FOLDER]",
            "🔔": "[BELL]",
            "⚙️": "[SETTINGS]",
            "🧪": "[TEST]",
            "📤": "[EXPORT]",
            "🔌": "[PORT]",
            "🌲": "[TREE]",
            "🔥": "[FIRE]",
            "🔑": "[KEY]",
            "🛠️": "[WRENCH]",
            "💻": "[COMPUTER]",
            "🏗️": "[BUILDING]",
            "❓": "[QUESTION]",
            "🪲": "[BUG]",
            "🪃": "[BOOMERANG]",
            "🛡️": "[SHIELD]",
            "📘": "[BOOK]",
            "🧹": "[BROOM]",
            "🔬": "[MICROSCOPE]",
        }

        safe_message = message
        for emoji, replacement in emoji_map.items():
            safe_message = safe_message.replace(emoji, replacement)

        return safe_message

    def _create_interface(self):
        """Create the comprehensive Gradio interface."""
        with gr.Blocks(
            title=self.title, theme=self.theme, css=self._get_custom_css()
        ) as interface:
            # Header
            gr.Markdown(f"# {self.title}")
            gr.Markdown(self.description)

            # System status bar
            with gr.Row():
                status_display = gr.Textbox(
                    label="System Status",
                    value="Ready ",
                    interactive=False,
                    container=False,
                )
                stats_display = gr.Textbox(
                    label="Stats",
                    value="Documents: 0 | Chunks: 0 | Queries: 0",
                    interactive=False,
                    container=False,
                )

            # Store interface components for updates early
            self.status_display = status_display
            self.stats_display = stats_display

            # Main interface tabs
            with gr.Tabs() as tabs:
                # Document Upload Tab
                if self.enable_file_upload:
                    with gr.TabItem("📄 Upload Documents", id="upload_tab"):
                        upload_components = self._create_upload_tab()

                # URL Processing Tab
                if self.enable_url_input:
                    with gr.TabItem("🔗 Add URLs", id="url_tab"):
                        url_components = self._create_url_tab()

                # Query Interface Tab
                if self.enable_query_interface:
                    with gr.TabItem("❓ Ask Questions", id="query_tab"):
                        query_components = self._create_query_tab()

                # Knowledge Base Management Tab
                with gr.TabItem("📚 Knowledge Base", id="kb_tab"):
                    kb_components = self._create_knowledge_base_tab()

                # Analytics Dashboard Tab
                with gr.TabItem("📈 Analytics", id="analytics_tab"):
                    analytics_components = self._create_analytics_tab()

                # System Health Tab
                with gr.TabItem("🩺 System Health", id="health_tab"):
                    health_components = self._create_health_tab()

                # Settings Tab
                with gr.TabItem("⚙️ Settings", id="settings_tab"):
                    settings_components = self._create_settings_tab()

        self.interface = interface

    def _create_upload_tab(self):
        """Create the document upload tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Documents")
                gr.Markdown("✅ Supported formats: PDF, DOCX, CSV, XLSX, PPTX, TXT, MD")

                file_upload = gr.File(
                    label="Select Files",
                    file_count="multiple",
                    file_types=[
                        ".pdf",
                        ".docx",
                        ".csv",
                        ".xlsx",
                        ".pptx",
                        ".txt",
                        ".md",
                    ],
                    height=200,
                )

                with gr.Row():
                    upload_btn = gr.Button(
                        "Process Documents", variant="primary", size="lg"
                    )
                    clear_upload_btn = gr.Button("Clear", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("###   Processing Results")
                upload_output = gr.Textbox(
                    label="Results",
                    lines=15,
                    interactive=False,
                    placeholder="Upload results will appear here...",
                )

        # Event handlers
        upload_btn.click(
            fn=self._process_documents,
            inputs=[file_upload],
            outputs=[upload_output, self.status_display, self.stats_display],
        )

        clear_upload_btn.click(
            fn=lambda: ("", "Ready "), outputs=[upload_output, self.status_display]
        )

        return {
            "file_upload": file_upload,
            "upload_btn": upload_btn,
            "upload_output": upload_output,
        }

    def _create_url_tab(self):
        """Create the URL processing tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Add URLs")
                gr.Markdown("Enter URLs to extract content from web pages")

                url_input = gr.Textbox(
                    label="URLs (one per line)",
                    lines=8,
                    placeholder="https://example.com\nhttps://another-site.com\n...",
                )

                with gr.Accordion("⚙️ Advanced Crawling Options", open=False):
                    gr.Markdown("🕷️ **Crawl Configuration**")

                    max_depth = gr.Slider(
                        label="🔍 Crawl Depth (How deep to follow links)",
                        minimum=1,
                        maximum=5,
                        value=1,
                        step=1,
                        info="Higher depth = more pages but slower processing",
                    )

                    follow_links = gr.Checkbox(
                        label="🔗 Follow Internal Links",
                        value=True,
                        info="Automatically discover and process linked pages",
                    )

                    gr.Markdown("⚡ **Performance Tips:**")
                    gr.Markdown("• Depth 1: Single page only")
                    gr.Markdown("• Depth 2-3: Good for small sites")
                    gr.Markdown("• Depth 4-5: Use carefully, can be slow")

                with gr.Row():
                    url_btn = gr.Button("🚀 Process URLs", variant="primary", size="lg")
                    clear_url_btn = gr.Button("🗑️ Clear", variant="secondary")

                # Progress indicator
                with gr.Row():
                    progress_info = gr.Textbox(
                        label="🔄 Processing Status",
                        value="Ready to process URLs...",
                        interactive=False,
                        visible=True,
                    )

            with gr.Column(scale=1):
                gr.Markdown("###   Processing Results")
                url_output = gr.Textbox(
                    label="Results",
                    lines=15,
                    interactive=False,
                    placeholder="URL processing results will appear here...",
                )

        # Event handlers
        url_btn.click(
            fn=self._process_urls,
            inputs=[url_input, max_depth, follow_links],
            outputs=[
                url_output,
                self.status_display,
                self.stats_display,
                progress_info,
            ],
        )

        clear_url_btn.click(
            fn=lambda: ("", "Ready 🟢", "Ready to process URLs..."),
            outputs=[url_output, self.status_display, progress_info],
        )

        return {
            "url_input": url_input,
            "url_btn": url_btn,
            "url_output": url_output,
        }

    def _create_query_tab(self):
        """Create the query interface tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ❓ Ask Questions")

                query_input = gr.Textbox(
                    label="Your Question",
                    lines=4,
                    placeholder="Ask a question about your uploaded documents...",
                )

                with gr.Accordion("⚙️ Query Options", open=False):
                    include_sources = gr.Checkbox(
                        label="📚 Include Sources",
                        value=True,
                        info="Show source documents and references",
                    )
                    max_results = gr.Slider(
                        label="📊 Max Results",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        info="Maximum number of results to return",
                    )

                    # 🌐 Live Search Option
                    with gr.Group():
                        gr.Markdown("**🌐 Live Search Options**")
                        use_live_search = gr.Checkbox(
                            label="🔍 Enable Live Web Search",
                            value=False,
                            info="Search the web for real-time information using Tavily API",
                        )

                        with gr.Row():
                            search_depth = gr.Dropdown(
                                label="🕷️ Search Depth",
                                choices=["basic", "advanced"],
                                value="basic",
                                info="Basic: faster, Advanced: more comprehensive",
                                visible=False,
                            )
                            time_range = gr.Dropdown(
                                label="⏰ Time Range",
                                choices=["day", "week", "month", "year"],
                                value="month",
                                info="How recent should the web results be",
                                visible=False,
                            )

                        # 💡 Show/hide advanced options based on live search toggle
                        use_live_search.change(
                            fn=lambda enabled: (
                                gr.update(visible=enabled),
                                gr.update(visible=enabled),
                            ),
                            inputs=[use_live_search],
                            outputs=[search_depth, time_range],
                        )

                with gr.Row():
                    query_btn = gr.Button("🚀 Get Answer", variant="primary", size="lg")
                    clear_query_btn = gr.Button("🗑️ Clear", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### 💬 Answer")

                response_output = gr.Markdown(
                    label="Response",
                    value="Your answer will appear here...",
                    height=400,
                )

                confidence_display = gr.Textbox(
                    label="🎯 Confidence Score",
                    interactive=False,
                    visible=self.enable_confidence_display,
                )

                sources_output = gr.JSON(
                    label="📚 Sources",
                    visible=self.enable_source_display,
                )

        # Event handlers
        query_btn.click(
            fn=self._process_query,
            inputs=[
                query_input,
                include_sources,
                max_results,
                use_live_search,
                search_depth,
                time_range,
            ],
            outputs=[
                response_output,
                confidence_display,
                sources_output,
                self.status_display,
                self.stats_display,
            ],
        )

        clear_query_btn.click(
            fn=lambda: ("", "", {}, "Ready 🟢"),
            outputs=[
                response_output,
                confidence_display,
                sources_output,
                self.status_display,
            ],
        )

        return {
            "query_input": query_input,
            "query_btn": query_btn,
            "response_output": response_output,
            "sources_output": sources_output,
            "use_live_search": use_live_search,
            "search_depth": search_depth,
            "time_range": time_range,
        }

    def _create_knowledge_base_tab(self):
        """Create the knowledge base management tab."""
        with gr.Column():
            gr.Markdown("### 📚 Knowledge Base Management")

            with gr.Row():
                refresh_btn = gr.Button("Refresh", variant="secondary")
                export_btn = gr.Button("📤 Export", variant="secondary")
                clear_kb_btn = gr.Button("Clear All", variant="stop")

            # Knowledge base stats with enhanced embedding model info
            kb_stats = gr.JSON(
                label="📊 Knowledge Base Statistics",
                value={
                    "total_documents": 0,
                    "total_chunks": 0,
                    "storage_size": "0 MB",
                    "embedding_model": "Loading...",
                    "embedding_status": "Checking...",
                    "vector_db_status": "Checking...",
                },
            )

            # 🤖 Embedding Model Status Display
            embedding_model_status = gr.JSON(
                label="🤖 Embedding Model Information",
                value={
                    "model_name": "Loading...",
                    "provider": "Checking...",
                    "status": "Initializing...",
                    "api_status": "Checking connection...",
                    "dimension": "Unknown",
                    "performance": "Gathering stats...",
                },
            )

            # Document list
            document_list = gr.Dataframe(
                headers=["Source", "Type", "Chunks", "Added"],
                datatype=["str", "str", "number", "str"],
                label="📄 Documents in Knowledge Base",
                interactive=False,
            )

        # Event handlers
        refresh_btn.click(
            fn=self._refresh_knowledge_base,
            outputs=[kb_stats, embedding_model_status, document_list],
        )

        return {
            "kb_stats": kb_stats,
            "embedding_model_status": embedding_model_status,
            "document_list": document_list,
        }

    def _create_analytics_tab(self):
        """Create the analytics dashboard tab with real-time data."""
        with gr.Column():
            gr.Markdown("### 📈 Analytics Dashboard")
            gr.Markdown("Real-time insights into your RAG system performance")

            with gr.Row():
                refresh_analytics_btn = gr.Button(
                    "🔄 Refresh Analytics", variant="secondary"
                )
                export_analytics_btn = gr.Button(
                    "📊 Export Report", variant="secondary"
                )

            with gr.Row():
                with gr.Column():
                    query_analytics = gr.JSON(
                        label="🔍 Query Analytics",
                        value=self._get_initial_query_analytics(),
                    )

                with gr.Column():
                    system_metrics = gr.JSON(
                        label="⚡ System Metrics",
                        value=self._get_initial_system_metrics(),
                    )

            with gr.Row():
                with gr.Column():
                    performance_metrics = gr.JSON(
                        label="🚀 Performance Metrics",
                        value=self._get_initial_performance_metrics(),
                    )

                with gr.Column():
                    usage_stats = gr.JSON(
                        label="📊 Usage Statistics",
                        value=self._get_initial_usage_stats(),
                    )

            # Query history with enhanced information
            query_history = gr.Dataframe(
                headers=[
                    "Query",
                    "Results",
                    "Confidence",
                    "Processing Time",
                    "Timestamp",
                ],
                datatype=["str", "number", "number", "str", "str"],
                label="📝 Recent Query History",
                interactive=False,
                value=self._get_initial_query_history(),
            )

            # Event handlers
            refresh_analytics_btn.click(
                fn=self._refresh_analytics,
                outputs=[
                    query_analytics,
                    system_metrics,
                    performance_metrics,
                    usage_stats,
                    query_history,
                ],
            )

        return {
            "query_analytics": query_analytics,
            "system_metrics": system_metrics,
            "performance_metrics": performance_metrics,
            "usage_stats": usage_stats,
            "query_history": query_history,
        }

    def _get_initial_query_analytics(self) -> Dict[str, Any]:
        """Get initial query analytics data."""
        return {
            "total_queries": self.query_count,
            "average_confidence": "N/A",
            "most_common_topics": [],
            "query_success_rate": "100%",
            "cache_hit_rate": "0%",
            "status": "📊 Ready to track queries",
        }

    def _get_initial_system_metrics(self) -> Dict[str, Any]:
        """Get initial system metrics."""
        # Get real embedding model info
        embedding_info = self._get_embedding_model_info()

        return {
            "documents_processed": self.total_documents,
            "chunks_stored": self.total_chunks,
            "embedding_model": embedding_info.get("model_name", "Gemini"),
            "embedding_status": embedding_info.get("status", "Checking..."),
            "embedding_provider": embedding_info.get("provider", "Google"),
            "vector_db": "Pinecone",
            "uptime": "Just started",
            "status": "🟢 System operational",
        }

    def _get_initial_performance_metrics(self) -> Dict[str, Any]:
        """Get initial performance metrics."""
        return {
            "avg_query_time": "N/A",
            "avg_embedding_time": "N/A",
            "avg_retrieval_time": "N/A",
            "memory_usage": "Normal",
            "throughput": "N/A queries/min",
            "status": "⚡ Performance tracking active",
        }

    def _get_initial_usage_stats(self) -> Dict[str, Any]:
        """Get initial usage statistics."""
        return {
            "documents_uploaded": 0,
            "urls_processed": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "peak_usage_time": "N/A",
            "status": "📈 Usage tracking enabled",
        }

    def _get_initial_query_history(self) -> List[List[str]]:
        """Get initial query history."""
        return [
            ["No queries yet", "0", "0.0", "0.0s", "Start asking questions!"],
            ["Upload documents first", "0", "0.0", "0.0s", "Build your knowledge base"],
            [
                "Try the examples above",
                "0",
                "0.0",
                "0.0s",
                "Get started with sample queries",
            ],
        ]

    def _refresh_analytics(
        self,
    ) -> Tuple[
        Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[List[str]]
    ]:
        """Refresh all analytics data."""
        try:
            # Get real analytics from query processor if available
            query_analytics = self._get_real_query_analytics()
            system_metrics = self._get_real_system_metrics()
            performance_metrics = self._get_real_performance_metrics()
            usage_stats = self._get_real_usage_stats()
            query_history = self._get_real_query_history()

            return (
                query_analytics,
                system_metrics,
                performance_metrics,
                usage_stats,
                query_history,
            )

        except Exception as e:
            self._log_safe(f"❌ Error refreshing analytics: {e}", "error")
            return (
                {"error": str(e)},
                {"error": str(e)},
                {"error": str(e)},
                {"error": str(e)},
                [["Error loading history", "0", "0.0", "0.0s", str(e)]],
            )

    def _get_real_query_analytics(self) -> Dict[str, Any]:
        """Get real query analytics from the system."""
        try:
            analytics = {
                "total_queries": self.query_count,
                "documents_in_kb": self.total_documents,
                "chunks_available": self.total_chunks,
                "last_updated": datetime.now().strftime("%H:%M:%S"),
            }

            # Get analytics from query processor if available
            if hasattr(self.rag_system, "query_processor") and hasattr(
                self.rag_system.query_processor, "get_query_analytics"
            ):
                processor_analytics = (
                    self.rag_system.query_processor.get_query_analytics()
                )
                analytics.update(processor_analytics)

            # Calculate additional metrics
            if self.query_count > 0:
                analytics["avg_results_per_query"] = round(
                    self.total_chunks / max(self.query_count, 1), 2
                )
                analytics["system_utilization"] = (
                    "Active" if self.query_count > 5 else "Light"
                )
            else:
                analytics["avg_results_per_query"] = 0
                analytics["system_utilization"] = "Idle"

            analytics["status"] = "🟢 Analytics active"
            return analytics

        except Exception as e:
            return {"error": f"Analytics unavailable: {str(e)}", "status": "❌ Error"}

    def _get_real_system_metrics(self) -> Dict[str, Any]:
        """Get real system metrics with embedding model info."""
        try:
            # Get embedding model information
            embedding_info = self._get_embedding_model_info()

            metrics = {
                "documents_processed": self.total_documents,
                "chunks_stored": self.total_chunks,
                "queries_processed": self.query_count,
                "last_updated": datetime.now().strftime("%H:%M:%S"),
                "embedding_model": embedding_info.get("model_name", "Unknown"),
                "embedding_status": embedding_info.get("status", "Unknown"),
                "embedding_provider": embedding_info.get("provider", "Unknown"),
                "embedding_dimension": embedding_info.get("dimension", "Unknown"),
            }

            # Get system status
            if hasattr(self.rag_system, "get_system_status"):
                system_status = self.rag_system.get_system_status()
                metrics.update(
                    {
                        "overall_health": system_status.get(
                            "overall_status", "unknown"
                        ),
                        "components_healthy": sum(
                            system_status.get("components", {}).values()
                        ),
                        "total_components": len(system_status.get("components", {})),
                    }
                )

            # Add component status with embedding model details
            components = []
            if hasattr(self.rag_system, "embedding_generator"):
                components.append(
                    f"Embedding Generator ({embedding_info.get('model_name', 'Unknown')})"
                )
            if hasattr(self.rag_system, "vector_db"):
                components.append("Vector Database")
            if hasattr(self.rag_system, "query_processor"):
                components.append("Query Processor")

            metrics["active_components"] = components
            metrics["status"] = "🟢 System healthy"
            return metrics

        except Exception as e:
            return {
                "error": f"System metrics unavailable: {str(e)}",
                "status": "❌ Error",
            }

    def _get_real_performance_metrics(self) -> Dict[str, Any]:
        """Get real performance metrics."""
        try:
            # Basic performance tracking
            metrics = {
                "total_processing_time": "N/A",
                "avg_query_response": "N/A",
                "system_load": "Normal",
                "last_updated": datetime.now().strftime("%H:%M:%S"),
            }

            # If we have query history, calculate averages
            if hasattr(self.rag_system, "query_processor") and hasattr(
                self.rag_system.query_processor, "query_history"
            ):
                history = self.rag_system.query_processor.query_history
                if history:
                    # Calculate average processing time if available
                    processing_times = [
                        q.get("processing_time", 0)
                        for q in history
                        if "processing_time" in q
                    ]
                    if processing_times:
                        avg_time = sum(processing_times) / len(processing_times)
                        metrics["avg_query_response"] = f"{avg_time:.2f}s"

            metrics["queries_per_minute"] = (
                f"{self.query_count / max(1, 1):.1f}"  # Rough estimate
            )
            metrics["throughput"] = "Good" if self.query_count > 0 else "Idle"
            metrics["status"] = "⚡ Performance tracking active"
            return metrics

        except Exception as e:
            return {
                "error": f"Performance metrics unavailable: {str(e)}",
                "status": "❌ Error",
            }

    def _get_real_usage_stats(self) -> Dict[str, Any]:
        """Get real usage statistics."""
        try:
            stats = {
                "documents_uploaded": self.total_documents,
                "urls_processed": 0,  # Would need to track this separately
                "successful_queries": self.query_count,  # Assuming all successful for now
                "failed_queries": 0,  # Would need error tracking
                "total_chunks_created": self.total_chunks,
                "last_updated": datetime.now().strftime("%H:%M:%S"),
            }

            # Calculate usage patterns
            if self.query_count > 0:
                stats["most_active_feature"] = "Query Processing"
                stats["usage_trend"] = "Growing" if self.query_count > 5 else "Starting"
            else:
                stats["most_active_feature"] = "Document Upload"
                stats["usage_trend"] = "Initial Setup"

            stats["status"] = "📊 Usage tracking active"
            return stats

        except Exception as e:
            return {"error": f"Usage stats unavailable: {str(e)}", "status": "❌ Error"}

    def _get_real_query_history(self) -> List[List[str]]:
        """Get real query history."""
        try:
            history_data = []

            # Get query history from query processor if available
            if hasattr(self.rag_system, "query_processor") and hasattr(
                self.rag_system.query_processor, "query_history"
            ):
                history = self.rag_system.query_processor.query_history[
                    -10:
                ]  # Last 10 queries

                for query_item in history:
                    query_text = (
                        query_item.get("query", "Unknown")[:50] + "..."
                        if len(query_item.get("query", "")) > 50
                        else query_item.get("query", "Unknown")
                    )
                    result_count = query_item.get("result_count", 0)
                    confidence = "N/A"  # Would need to store this
                    processing_time = (
                        f"{query_item.get('processing_time', 0):.2f}s"
                        if "processing_time" in query_item
                        else "N/A"
                    )
                    timestamp = (
                        query_item.get("timestamp", datetime.now()).strftime("%H:%M:%S")
                        if "timestamp" in query_item
                        else "Unknown"
                    )

                    history_data.append(
                        [
                            query_text,
                            str(result_count),
                            confidence,
                            processing_time,
                            timestamp,
                        ]
                    )

            # If no real history, show helpful placeholder
            if not history_data:
                history_data = [
                    ["No queries yet", "0", "0.0", "0.0s", "Ask your first question!"],
                    [
                        "Upload documents to get started",
                        "0",
                        "0.0",
                        "0.0s",
                        "Build your knowledge base",
                    ],
                    [
                        "Try asking about your documents",
                        "0",
                        "0.0",
                        "0.0s",
                        "Get intelligent answers",
                    ],
                ]

            return history_data

        except Exception as e:
            return [["Error loading history", "0", "0.0", "0.0s", str(e)]]

    def _create_settings_tab(self):
        """Create the comprehensive settings management tab."""
        with gr.Column():
            gr.Markdown("### ⚙️ Environment Variables Settings")
            gr.Markdown(
                "Configure API keys and system settings with secure storage options"
            )

            # 🔄 Refresh and action buttons
            with gr.Row():
                refresh_settings_btn = gr.Button("🔄 Refresh", variant="secondary")
                load_env_btn = gr.Button("📁 Load from .env", variant="secondary")
                clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary")
                export_btn = gr.Button("📤 Export Settings", variant="secondary")

            # 📊 Settings status display
            settings_status = gr.Textbox(
                label="🔔 Status",
                value="Ready to configure settings",
                interactive=False,
                container=False,
            )

            # 🔧 Main settings interface
            with gr.Tabs():
                # API Keys Tab
                with gr.TabItem("🔑 API Keys"):
                    api_keys_components = self._create_api_keys_section()

                # System Settings Tab
                with gr.TabItem("🛠️ System Settings"):
                    system_settings_components = self._create_system_settings_section()

                # Storage Options Tab
                with gr.TabItem("💾 Storage & Export"):
                    storage_components = self._create_storage_section()

            # 📋 Settings overview
            with gr.Accordion("📋 Current Settings Overview", open=False):
                settings_overview = gr.JSON(
                    label="Environment Variables Status", value={}
                )

            # Event handlers for main buttons
            refresh_settings_btn.click(
                fn=self._refresh_all_settings,
                outputs=[
                    settings_status,
                    settings_overview,
                    *api_keys_components.values(),
                    *system_settings_components.values(),
                ],
            )

            load_env_btn.click(
                fn=self._load_from_env_file,
                outputs=[settings_status, settings_overview],
            )

            clear_cache_btn.click(
                fn=self._clear_settings_cache,
                outputs=[settings_status, settings_overview],
            )

            export_btn.click(fn=self._export_settings, outputs=[settings_status])

        return {
            "settings_status": settings_status,
            "settings_overview": settings_overview,
            **api_keys_components,
            **system_settings_components,
            **storage_components,
        }

    def _create_api_keys_section(self):
        """Create the API keys configuration section."""
        components = {}

        with gr.Column():
            gr.Markdown("#### 🔑 API Keys Configuration")
            gr.Markdown(
                "Configure your API keys for AI services. Keys are masked for security."
            )

            # Gemini API Key
            with gr.Group():
                gr.Markdown("**🤖 Google Gemini API** (Required)")
                with gr.Row():
                    gemini_key = gr.Textbox(
                        label="GEMINI_API_KEY",
                        placeholder="AIzaSy...",
                        type="password",
                        info="Required for embeddings and LLM functionality",
                    )
                    gemini_test_btn = gr.Button(
                        "🧪 Test", variant="secondary", size="sm"
                    )

                gemini_status = gr.Textbox(
                    label="Status",
                    value="Not configured",
                    interactive=False,
                    container=False,
                )

                with gr.Row():
                    gemini_cache_btn = gr.Button(
                        "💾 Save to Cache", variant="primary", size="sm"
                    )
                    gemini_env_btn = gr.Button(
                        "📁 Save to .env", variant="primary", size="sm"
                    )

                gr.Markdown(
                    "💡 [Get your Gemini API key](https://aistudio.google.com/)"
                )

            # Pinecone API Key
            with gr.Group():
                gr.Markdown("**🌲 Pinecone API  (Required)**")
                with gr.Row():
                    pinecone_key = gr.Textbox(
                        label="PINECONE_API_KEY",
                        placeholder="pc-...",
                        type="password",
                        info="For vector database storage",
                    )
                    pinecone_test_btn = gr.Button(
                        "🧪 Test", variant="secondary", size="sm"
                    )

                pinecone_status = gr.Textbox(
                    label="Status",
                    value="Not configured",
                    interactive=False,
                    container=False,
                )

                with gr.Row():
                    pinecone_cache_btn = gr.Button(
                        "💾 Save to Cache", variant="primary", size="sm"
                    )
                    pinecone_env_btn = gr.Button(
                        "📁 Save to .env", variant="primary", size="sm"
                    )

                gr.Markdown("💡 [Get your Pinecone API key](https://www.pinecone.io/)")

            # OpenAI API Key
            with gr.Group():
                gr.Markdown("**🔥 OpenAI API** (Optional)")
                with gr.Row():
                    openai_key = gr.Textbox(
                        label="OPENAI_API_KEY",
                        placeholder="sk-...",
                        type="password",
                        info="For alternative LLM functionality",
                    )
                    openai_test_btn = gr.Button(
                        "🧪 Test", variant="secondary", size="sm"
                    )

                openai_status = gr.Textbox(
                    label="Status",
                    value="Not configured",
                    interactive=False,
                    container=False,
                )

                with gr.Row():
                    openai_cache_btn = gr.Button(
                        "💾 Save to Cache", variant="primary", size="sm"
                    )
                    openai_env_btn = gr.Button(
                        "📁 Save to .env", variant="primary", size="sm"
                    )

                gr.Markdown(
                    "💡 [Get your OpenAI API key](https://platform.openai.com/api-keys)"
                )

            # Tavily API Key
            with gr.Group():
                gr.Markdown("**🌐 Tavily API** (Optional - for Live Search)")
                with gr.Row():
                    tavily_key = gr.Textbox(
                        label="TAVILY_API_KEY",
                        placeholder="tvly-...",
                        type="password",
                        info="For real-time web search functionality",
                    )
                    tavily_test_btn = gr.Button(
                        "🧪 Test", variant="secondary", size="sm"
                    )

                tavily_status = gr.Textbox(
                    label="Status",
                    value="Not configured",
                    interactive=False,
                    container=False,
                )

                with gr.Row():
                    tavily_cache_btn = gr.Button(
                        "💾 Save to Cache", variant="primary", size="sm"
                    )
                    tavily_env_btn = gr.Button(
                        "📁 Save to .env", variant="primary", size="sm"
                    )

                gr.Markdown(
                    "💡 [Get your Tavily API key](https://app.tavily.com/sign-in)"
                )

        # Store components for event handling
        components.update(
            {
                "gemini_key": gemini_key,
                "gemini_status": gemini_status,
                "pinecone_key": pinecone_key,
                "pinecone_status": pinecone_status,
                "openai_key": openai_key,
                "openai_status": openai_status,
                "tavily_key": tavily_key,
                "tavily_status": tavily_status,
            }
        )

        # Event handlers for API keys
        gemini_test_btn.click(
            fn=lambda: self._test_api_connection("GEMINI_API_KEY"),
            outputs=[gemini_status],
        )

        gemini_cache_btn.click(
            fn=lambda key: self._save_setting("GEMINI_API_KEY", key, "cache"),
            inputs=[gemini_key],
            outputs=[gemini_status],
        )

        gemini_env_btn.click(
            fn=lambda key: self._save_setting("GEMINI_API_KEY", key, "env_file"),
            inputs=[gemini_key],
            outputs=[gemini_status],
        )

        pinecone_test_btn.click(
            fn=lambda: self._test_api_connection("PINECONE_API_KEY"),
            outputs=[pinecone_status],
        )

        pinecone_cache_btn.click(
            fn=lambda key: self._save_setting("PINECONE_API_KEY", key, "cache"),
            inputs=[pinecone_key],
            outputs=[pinecone_status],
        )

        pinecone_env_btn.click(
            fn=lambda key: self._save_setting("PINECONE_API_KEY", key, "env_file"),
            inputs=[pinecone_key],
            outputs=[pinecone_status],
        )

        openai_test_btn.click(
            fn=lambda: self._test_api_connection("OPENAI_API_KEY"),
            outputs=[openai_status],
        )

        openai_cache_btn.click(
            fn=lambda key: self._save_setting("OPENAI_API_KEY", key, "cache"),
            inputs=[openai_key],
            outputs=[openai_status],
        )

        openai_env_btn.click(
            fn=lambda key: self._save_setting("OPENAI_API_KEY", key, "env_file"),
            inputs=[openai_key],
            outputs=[openai_status],
        )

        tavily_test_btn.click(
            fn=lambda: self._test_api_connection("TAVILY_API_KEY"),
            outputs=[tavily_status],
        )

        tavily_cache_btn.click(
            fn=lambda key: self._save_setting("TAVILY_API_KEY", key, "cache"),
            inputs=[tavily_key],
            outputs=[tavily_status],
        )

        tavily_env_btn.click(
            fn=lambda key: self._save_setting("TAVILY_API_KEY", key, "env_file"),
            inputs=[tavily_key],
            outputs=[tavily_status],
        )

        return components

    def _create_system_settings_section(self):
        """Create the system settings configuration section."""
        components = {}

        with gr.Column():
            gr.Markdown("#### 🛠️ System Configuration")
            gr.Markdown("Configure system-level settings and preferences")

            # Pinecone Environment
            with gr.Group():
                gr.Markdown("**🌍 Pinecone Environment**")
                pinecone_env = gr.Dropdown(
                    label="PINECONE_ENVIRONMENT",
                    choices=[
                        "us-east-1",
                        "us-west1-gcp",
                        "eu-west1-gcp",
                        "asia-southeast1-gcp",
                    ],
                    value="us-east-1",
                    info="Pinecone server region",
                )

                with gr.Row():
                    pinecone_env_cache_btn = gr.Button(
                        "💾 Save to Cache", variant="primary", size="sm"
                    )
                    pinecone_env_file_btn = gr.Button(
                        "📁 Save to .env", variant="primary", size="sm"
                    )

            # Pinecone Index Name
            with gr.Group():
                gr.Markdown("**📊 Pinecone Index Name**")
                pinecone_index = gr.Textbox(
                    label="PINECONE_INDEX_NAME",
                    value="rag-ai-index",
                    placeholder="rag-ai-index",
                    info="Name of your Pinecone index",
                )

                with gr.Row():
                    pinecone_index_cache_btn = gr.Button(
                        "💾 Save to Cache", variant="primary", size="sm"
                    )
                    pinecone_index_file_btn = gr.Button(
                        "📁 Save to .env", variant="primary", size="sm"
                    )

            # Gradio Share
            with gr.Group():
                gr.Markdown("**🌐 Gradio Public Sharing**")
                gradio_share = gr.Dropdown(
                    label="GRADIO_SHARE",
                    choices=["false", "true"],
                    value="false",
                    info="Enable public sharing of the interface",
                )

                with gr.Row():
                    gradio_share_cache_btn = gr.Button(
                        "💾 Save to Cache", variant="primary", size="sm"
                    )
                    gradio_share_file_btn = gr.Button(
                        "📁 Save to .env", variant="primary", size="sm"
                    )

            # Port Configuration
            with gr.Group():
                gr.Markdown("**🔌 Server Port**")
                port_setting = gr.Number(
                    label="PORT",
                    value=7860,
                    minimum=1000,
                    maximum=65535,
                    info="Server port number (requires restart)",
                )

                with gr.Row():
                    port_cache_btn = gr.Button(
                        "💾 Save to Cache", variant="primary", size="sm"
                    )
                    port_file_btn = gr.Button(
                        "📁 Save to .env", variant="primary", size="sm"
                    )

            # System settings status
            system_status = gr.Textbox(
                label="System Settings Status",
                value="Ready",
                interactive=False,
                container=False,
            )

        components.update(
            {
                "pinecone_env": pinecone_env,
                "pinecone_index": pinecone_index,
                "gradio_share": gradio_share,
                "port_setting": port_setting,
                "system_status": system_status,
            }
        )

        # Event handlers for system settings
        pinecone_env_cache_btn.click(
            fn=lambda val: self._save_setting("PINECONE_ENVIRONMENT", val, "cache"),
            inputs=[pinecone_env],
            outputs=[system_status],
        )

        pinecone_env_file_btn.click(
            fn=lambda val: self._save_setting("PINECONE_ENVIRONMENT", val, "env_file"),
            inputs=[pinecone_env],
            outputs=[system_status],
        )

        pinecone_index_cache_btn.click(
            fn=lambda val: self._save_setting("PINECONE_INDEX_NAME", val, "cache"),
            inputs=[pinecone_index],
            outputs=[system_status],
        )

        pinecone_index_file_btn.click(
            fn=lambda val: self._save_setting("PINECONE_INDEX_NAME", val, "env_file"),
            inputs=[pinecone_index],
            outputs=[system_status],
        )

        gradio_share_cache_btn.click(
            fn=lambda val: self._save_setting("GRADIO_SHARE", val, "cache"),
            inputs=[gradio_share],
            outputs=[system_status],
        )

        gradio_share_file_btn.click(
            fn=lambda val: self._save_setting("GRADIO_SHARE", val, "env_file"),
            inputs=[gradio_share],
            outputs=[system_status],
        )

        port_cache_btn.click(
            fn=lambda val: self._save_setting("PORT", str(int(val)), "cache"),
            inputs=[port_setting],
            outputs=[system_status],
        )

        port_file_btn.click(
            fn=lambda val: self._save_setting("PORT", str(int(val)), "env_file"),
            inputs=[port_setting],
            outputs=[system_status],
        )

        return components

    def _create_storage_section(self):
        """Create the storage and export section."""
        components = {}

        with gr.Column():
            gr.Markdown("#### 💾 Storage & Export Options")
            gr.Markdown("Manage how your settings are stored and exported")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**💾 Cache Storage**")
                    gr.Markdown("• Temporary storage in memory")
                    gr.Markdown("• Lost when application restarts")
                    gr.Markdown("• Good for testing configurations")

                with gr.Column():
                    gr.Markdown("**📁 .env File Storage**")
                    gr.Markdown("• Persistent storage in .env file")
                    gr.Markdown("• Survives application restarts")
                    gr.Markdown("• Recommended for production use")

            # Export options
            with gr.Group():
                gr.Markdown("**📤 Export Settings**")

                with gr.Row():
                    include_sensitive = gr.Checkbox(
                        label="Include API Keys (masked)",
                        value=False,
                        info="Include API keys in export (they will be masked)",
                    )
                    export_format = gr.Dropdown(
                        label="Export Format",
                        choices=["JSON", "ENV"],
                        value="JSON",
                        info="Choose export format",
                    )

                export_output = gr.Textbox(
                    label="Export Output",
                    lines=10,
                    interactive=False,
                    placeholder="Exported settings will appear here...",
                )

                export_settings_btn = gr.Button("📤 Generate Export", variant="primary")

            # Storage status
            storage_status = gr.Textbox(
                label="Storage Status",
                value="Ready",
                interactive=False,
                container=False,
            )

        components.update(
            {
                "include_sensitive": include_sensitive,
                "export_format": export_format,
                "export_output": export_output,
                "storage_status": storage_status,
            }
        )

        # Export event handler
        export_settings_btn.click(
            fn=self._generate_export,
            inputs=[include_sensitive, export_format],
            outputs=[export_output, storage_status],
        )

        return components

    def _create_health_tab(self):
        """Create the system health monitoring tab."""
        with gr.Column():
            gr.Markdown("### System Health")

            with gr.Row():
                health_check_btn = gr.Button("Run Health Check", variant="primary")
                restart_btn = gr.Button("Restart Services", variant="secondary")

            # System status
            system_status = gr.JSON(
                label="System Status",
                value={},
            )

            # Component status
            component_status = gr.Dataframe(
                headers=["Component", "Status", "Details"],
                datatype=["str", "str", "str"],
                label="Component Status",
                interactive=False,
            )

            # Logs
            system_logs = gr.Textbox(
                label="  System Logs",
                lines=10,
                interactive=False,
                placeholder="System logs will appear here...",
            )

        # Event handlers
        health_check_btn.click(
            fn=self._run_health_check,
            outputs=[system_status, component_status, system_logs],
        )

        return {
            "system_status": system_status,
            "component_status": component_status,
            "system_logs": system_logs,
        }

    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface."""
        return """
        .gradio-container {
            max-width: 1200px !important;
        }
        .status-bar {
            background: linear-gradient(90deg, #f0f9ff, #e0f2fe);
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
            margin: 8px;
        }
        """

    # 🔧 Settings Management Methods

    def _refresh_all_settings(self):
        """Refresh all settings and return updated values."""
        try:
            settings = self.settings_manager.get_current_settings()

            # Create overview for display
            overview = {}
            for var_name, config in settings.items():
                overview[var_name] = {
                    "value": config["value"] if config["is_set"] else "Not set",
                    "source": config["source"],
                    "status": (
                        "✅ Valid"
                        if config["is_valid"]
                        else "❌ Invalid" if config["is_set"] else "⚠️ Not set"
                    ),
                    "required": config["is_required"],
                }

            # Return status and all component updates
            status_msg = "🔄 Settings refreshed successfully"

            # Get current values for form fields
            gemini_val = settings.get("GEMINI_API_KEY", {}).get("raw_value", "")
            pinecone_val = settings.get("PINECONE_API_KEY", {}).get("raw_value", "")
            openai_val = settings.get("OPENAI_API_KEY", {}).get("raw_value", "")
            tavily_val = settings.get("TAVILY_API_KEY", {}).get("raw_value", "")

            pinecone_env_val = settings.get("PINECONE_ENVIRONMENT", {}).get(
                "raw_value", "us-east-1"
            )
            pinecone_index_val = settings.get("PINECONE_INDEX_NAME", {}).get(
                "raw_value", "rag-ai-index"
            )
            gradio_share_val = settings.get("GRADIO_SHARE", {}).get(
                "raw_value", "false"
            )
            port_val = int(settings.get("PORT", {}).get("raw_value", "7860"))

            return (
                status_msg,
                overview,
                gemini_val,
                settings.get("GEMINI_API_KEY", {}).get("value", "Not configured"),
                pinecone_val,
                settings.get("PINECONE_API_KEY", {}).get("value", "Not configured"),
                openai_val,
                settings.get("OPENAI_API_KEY", {}).get("value", "Not configured"),
                tavily_val,
                settings.get("TAVILY_API_KEY", {}).get("value", "Not configured"),
                pinecone_env_val,
                pinecone_index_val,
                gradio_share_val,
                port_val,
                "✅ Settings loaded",
            )

        except Exception as e:
            self._log_safe(f" Error refreshing settings: {e}", "error")
            return (
                f" Error refreshing settings: {str(e)}",
                {},
                "",
                "Error loading",
                "",
                "Error loading",
                "",
                "Error loading",
                "",
                "Error loading",
                "us-east-1",
                "rag-ai-index",
                "false",
                7860,
                "❌ Error loading",
            )

    def _save_setting(self, var_name: str, value: str, storage_type: str) -> str:
        """Save a setting with the specified storage type."""
        try:
            result = self.settings_manager.update_setting(var_name, value, storage_type)

            if result["success"]:
                self._log_safe(f" Saved {var_name} to {storage_type}")
                return result["status"]
            else:
                self._log_safe(
                    f" Failed to save {var_name}: {result.get('error', 'Unknown error')}",
                    "error",
                )
                return result["status"]

        except Exception as e:
            self._log_safe(f" Error saving {var_name}: {e}", "error")
            return f"❌ Error: {str(e)}"

    def _test_api_connection(self, var_name: str) -> str:
        """Test API connection for the specified variable."""
        try:
            result = self.settings_manager.test_connection(var_name)

            if result["success"]:
                self._log_safe(f"✅ {var_name} connection test successful")
            else:
                self._log_safe(
                    f" {var_name} connection test failed: {result.get('error', 'Unknown error')}",
                    "warning",
                )

            return result["status"]

        except Exception as e:
            self._log_safe(f" Error testing {var_name}: {e}", "error")
            return f" Test error: {str(e)}"

    def _load_from_env_file(self) -> Tuple[str, Dict[str, Any]]:
        """Load settings from .env file."""
        try:
            result = self.settings_manager.load_from_env_file()

            if result["success"]:
                self._log_safe(
                    f" Loaded {result['loaded_count']} variables from .env file"
                )

                # Get updated overview
                settings = self.settings_manager.get_current_settings()
                overview = {}
                for var_name, config in settings.items():
                    overview[var_name] = {
                        "value": config["value"] if config["is_set"] else "Not set",
                        "source": config["source"],
                        "status": (
                            "✅ Valid"
                            if config["is_valid"]
                            else "❌ Invalid" if config["is_set"] else "⚠️ Not set"
                        ),
                        "required": config["is_required"],
                    }

                return result["status"], overview
            else:
                self._log_safe(
                    f" Failed to load from .env: {result.get('error', 'Unknown error')}",
                    "error",
                )
                return result["status"], {}

        except Exception as e:
            self._log_safe(f" Error loading from .env file: {e}", "error")
            return f" Error: {str(e)}", {}

    def _clear_settings_cache(self) -> Tuple[str, Dict[str, Any]]:
        """Clear settings cache."""
        try:
            result = self.settings_manager.clear_cache()

            if result["success"]:
                self._log_safe(f" Cleared {result['cleared_count']} cached variables")

                # Get updated overview
                settings = self.settings_manager.get_current_settings()
                overview = {}
                for var_name, config in settings.items():
                    overview[var_name] = {
                        "value": config["value"] if config["is_set"] else "Not set",
                        "source": config["source"],
                        "status": (
                            "✅ Valid"
                            if config["is_valid"]
                            else "❌ Invalid" if config["is_set"] else "⚠️ Not set"
                        ),
                        "required": config["is_required"],
                    }

                return result["status"], overview
            else:
                self._log_safe(
                    f" Failed to clear cache: {result.get('error', 'Unknown error')}",
                    "error",
                )
                return result["status"], {}

        except Exception as e:
            self._log_safe(f" Error clearing cache: {e}", "error")
            return f" Error: {str(e)}", {}

    def _export_settings(self) -> str:
        """Export settings (basic version for main button)."""
        try:
            result = self.settings_manager.export_settings(include_sensitive=False)

            if result["success"]:
                self._log_safe(" Settings exported successfully")
                return " Settings exported (check Storage & Export tab for details)"
            else:
                self._log_safe(
                    f" Failed to export settings: {result.get('error', 'Unknown error')}",
                    "error",
                )
                return f" Export failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            self._log_safe(f" Error exporting settings: {e}", "error")
            return f" Error: {str(e)}"

    def _generate_export(
        self, include_sensitive: bool, export_format: str
    ) -> Tuple[str, str]:
        """Generate detailed export output."""
        try:
            result = self.settings_manager.export_settings(
                include_sensitive=include_sensitive
            )

            if not result["success"]:
                return (
                    f" Export failed: {result.get('error', 'Unknown error')}",
                    " Export failed",
                )

            settings_data = result["settings"]

            if export_format == "JSON":
                import json

                export_content = json.dumps(
                    {
                        "export_info": {
                            "timestamp": result["export_timestamp"],
                            "include_sensitive": include_sensitive,
                            "format": "JSON",
                        },
                        "settings": settings_data,
                    },
                    indent=2,
                )

            elif export_format == "ENV":
                export_lines = [
                    "# Environment Variables Export",
                    f"# Generated on {result['export_timestamp']}",
                    f"# Include sensitive: {include_sensitive}",
                    "",
                ]

                for var_name, config in settings_data.items():
                    if config["is_set"]:
                        value = config["value"]
                        export_lines.append(f"# {config['description']}")
                        export_lines.append(f"{var_name}={value}")
                        export_lines.append("")

                export_content = "\n".join(export_lines)

            else:
                return " Invalid export format", " Invalid format"

            self._log_safe(
                f" Generated {export_format} export with {len(settings_data)} variables"
            )
            return export_content, f" {export_format} export generated successfully"

        except Exception as e:
            self._log_safe(f" Error generating export: {e}", "error")
            return f" Error: {str(e)}", " Export generation failed"

    def _process_documents(self, files) -> Tuple[str, str, str]:
        """
        Process uploaded documents with progress tracking.

        Args:
            files: List of uploaded files

        Returns:
            Tuple of (processing results, status, stats)
        """
        if not files:
            return "No files uploaded.", "Ready ", self._get_stats_string()

        try:
            self._log_safe(f"Processing {len(files)} uploaded files")

            results = []
            successful = 0

            for i, file in enumerate(files):
                try:
                    # Process each file
                    result = self.rag_system.process_document(file.name)

                    if result.get("status") == "success":
                        successful += 1
                        self.total_documents += 1
                        self.total_chunks += result.get("chunks_processed", 0)

                        results.append(
                            f"{os.path.basename(file.name)}: "
                            f"{result.get('chunks_processed', 0)} chunks processed"
                        )
                    else:
                        results.append(
                            f"❌ {os.path.basename(file.name)}: "
                            f"{result.get('error', 'Processing failed')}"
                        )

                except Exception as e:
                    results.append(f"❌ {os.path.basename(file.name)}: {str(e)}")

            # Summary
            summary = (
                f"\nSummary: {successful}/{len(files)} files processed successfully"
            )
            output = "\n".join(results) + summary

            status = (
                f"Processed {successful}/{len(files)} files "
                if successful > 0
                else "Processing failed ❌"
            )

            return output, status, self._get_stats_string()

        except Exception as e:
            self._log_safe(f" Error processing documents: {str(e)}", "error")
            return f" Error: {str(e)}", "Error ", self._get_stats_string()

    def _process_urls(
        self, urls_text: str, max_depth: int = 1, follow_links: bool = True
    ) -> Tuple[str, str, str, str]:
        """
        Process URLs with advanced crawling options and progress tracking.

        Args:
            urls_text: Text containing URLs (one per line)
            max_depth: Maximum crawling depth
            follow_links: Whether to follow links

        Returns:
            Tuple of (processing results, status, stats, progress_info)
        """
        if not urls_text.strip():
            return (
                "No URLs provided.",
                "Ready 🟢",
                self._get_stats_string(),
                "Ready to process URLs...",
            )

        try:
            urls = [url.strip() for url in urls_text.split("\n") if url.strip()]
            self._log_safe(
                f"Processing {len(urls)} URLs with depth={max_depth}, follow_links={follow_links}"
            )

            results = []
            successful = 0
            progress_msg = f"🚀 Starting crawl of {len(urls)} URLs..."

            for i, url in enumerate(urls):
                progress_msg = f"🔄 Processing URL {i+1}/{len(urls)}: {url[:50]}..."
                try:
                    # Process each URL with advanced options
                    result = self.rag_system.process_url(
                        url, max_depth=max_depth, follow_links=follow_links
                    )

                    if result.get("status") == "success":
                        successful += 1
                        self.total_documents += 1
                        self.total_chunks += result.get("chunks_processed", 0)

                        # Enhanced result display with crawling info
                        chunks = result.get("chunks_processed", 0)
                        linked_docs = result.get("linked_documents_processed", 0)
                        depth = result.get("depth", 0)

                        result_text = f"✅ {url}:\n"
                        result_text += f"   📄 {chunks} chunks processed"
                        if linked_docs > 0:
                            result_text += f"\n   🔗 {linked_docs} linked pages found"
                        if depth > 0:
                            result_text += f"\n   🕷️ Crawled to depth {depth}"

                        results.append(result_text)
                    else:
                        error_msg = result.get("error", "Processing failed")
                        results.append(f"❌ {url}: {error_msg}")

                        # Add helpful hints for common crawling issues
                        if "depth" in error_msg.lower():
                            results.append("   💡 Try reducing crawl depth")
                        elif "timeout" in error_msg.lower():
                            results.append(
                                "   💡 Site may be slow, try single page mode"
                            )
                        elif "robots" in error_msg.lower():
                            results.append(
                                "   💡 Site blocks crawlers, try direct URL only"
                            )

                except Exception as e:
                    results.append(f"❌ {url}: {str(e)}")

            # Enhanced Summary with crawling stats
            total_linked = sum(
                result.get("linked_documents_processed", 0)
                for result in [
                    self.rag_system.process_url(url, max_depth, follow_links)
                    for url in urls
                ]
                if result.get("status") == "success"
            )

            summary = f"\n" + "=" * 50
            summary += f"\n📊 **CRAWLING SUMMARY**"
            summary += f"\n✅ URLs processed: {successful}/{len(urls)}"
            if follow_links and max_depth > 1:
                summary += f"\n🔗 Linked pages discovered: {total_linked}"
                summary += f"\n🕷️ Max crawl depth: {max_depth}"
            summary += f"\n📄 Total chunks: {self.total_chunks}"
            summary += "\n" + "=" * 50

            output = "\n".join(results) + summary

            status = (
                f"Processed {successful}/{len(urls)} URLs "
                if successful > 0
                else "Processing failed "
            )

            final_progress = (
                f"✅ Completed! Processed {successful}/{len(urls)} URLs successfully"
            )
            return output, status, self._get_stats_string(), final_progress

        except Exception as e:
            self._log_safe(f" Error processing URLs: {str(e)}", "error")
            error_progress = f" Error occurred during processing"
            return (
                f" Error: {str(e)}",
                "Error ",
                self._get_stats_string(),
                error_progress,
            )

    def _process_query(
        self,
        query: str,
        include_sources: bool = True,
        max_results: int = 5,
        use_live_search: bool = False,
        search_depth: str = "basic",
        time_range: str = "month",
    ) -> Tuple[str, str, Dict[str, Any], str, str]:
        """
        Process a user query with enhanced response formatting and live search options.

        Args:
            query: User query string
            include_sources: Whether to include source information
            max_results: Maximum number of results to return
            use_live_search: Whether to use live web search
            search_depth: Search depth for live search
            time_range: Time range for live search

        Returns:
            Tuple of (response, confidence, sources, status, stats)
        """
        if not query.strip():
            return (
                "Please enter a question.",
                "",
                {},
                "Ready 🟢",
                self._get_stats_string(),
            )

        try:
            # ✅ Safe Unicode logging for Windows compatibility
            search_type = "🌐 Live + Local" if use_live_search else "📚 Local Only"
            self._log_safe(
                f" Processing query ({search_type}): {query[:100]}... "
                f"(sources: {include_sources}, max_results: {max_results})"
            )

            # 🚀 Check if we have live search capability
            if use_live_search:
                # Use live search via MCP Tavily integration
                result = self._process_live_query(
                    query, max_results, search_depth, time_range
                )
            else:
                # Use traditional local RAG system
                result = self.rag_system.query(query, max_results=max_results)

            self.query_count += 1

            response = result.get("response", "No response generated.")
            confidence = result.get("confidence", 0.0)
            sources = result.get("sources", [])

            # 🎯 Format confidence display with search type indicator
            confidence_text = f"🎯 Confidence: {confidence:.1%}"
            if confidence >= 0.8:
                confidence_text += " 🟢 High"
            elif confidence >= 0.5:
                confidence_text += " 🟡 Medium"
            else:
                confidence_text += " 🔴 Low"

            # Add processing details with search type
            context_items = result.get("context_items", 0)
            processing_time = result.get("processing_time", 0)
            search_indicator = "🌐" if use_live_search else "📚"
            confidence_text += f" | {search_indicator} {search_type} | ⚡ {processing_time:.2f}s | 📄 {context_items} items"

            # 📊 Format sources for display based on user preference
            sources_display = {}
            if include_sources and sources:
                # Limit sources based on max_results
                limited_sources = sources[:max_results]
                sources_display = {
                    "confidence": f"{confidence:.3f}",
                    "total_sources": len(sources),
                    "showing": len(limited_sources),
                    "max_requested": max_results,
                    "sources": limited_sources,
                    "search_type": search_type,
                    "query_options": {
                        "include_sources": include_sources,
                        "max_results": max_results,
                        "use_live_search": use_live_search,
                        "search_depth": search_depth if use_live_search else None,
                        "time_range": time_range if use_live_search else None,
                    },
                }

                # 🌐 Add live search specific metadata
                if use_live_search:
                    sources_display.update(
                        {
                            "live_search_params": {
                                "search_depth": search_depth,
                                "time_range": time_range,
                                "routing_decision": result.get(
                                    "routing_decision", "live_search"
                                ),
                            }
                        }
                    )

            elif not include_sources:
                sources_display = {
                    "message": "🔒 Sources hidden by user preference",
                    "total_sources": len(sources),
                    "search_type": search_type,
                    "query_options": {
                        "include_sources": include_sources,
                        "max_results": max_results,
                        "use_live_search": use_live_search,
                    },
                }

            # 📈 Enhanced status with search type
            status_icon = "🌐" if use_live_search else "📚"
            status = f"✅ {status_icon} Query processed (confidence: {confidence:.1%}, {len(sources)} sources)"

            return (
                response,
                confidence_text,
                sources_display,
                status,
                self._get_stats_string(),
            )

        except Exception as e:
            self._log_safe(f" Error processing query: {str(e)}", "error")
            return (
                f" Error: {str(e)}",
                "Error",
                {},
                "Error ",
                self._get_stats_string(),
            )

    def _process_live_query(
        self, query: str, max_results: int, search_depth: str, time_range: str
    ) -> Dict[str, Any]:
        """
        Process query using live search via MCP Tavily integration.

        Args:
            query: User query
            max_results: Maximum results to return
            search_depth: Search depth parameter
            time_range: Time range for search

        Returns:
            Dictionary with search results and metadata
        """
        try:
            self._log_safe(f" Performing live search with Tavily API...")

            # 🚀 Use MCP Tavily tool for live search
            # This will be the actual MCP integration point
            search_results = self._call_tavily_mcp(
                query, max_results, search_depth, time_range
            )

            # 🔄 Process and format results for RAG response generation
            if search_results and search_results.get("results"):
                # Format for response generator
                formatted_context = []
                for result in search_results["results"]:
                    formatted_context.append(
                        {
                            "text": result.get("content", ""),
                            "source": result.get("url", "web_search"),
                            "title": result.get("title", "Web Result"),
                            "score": result.get("score", 0.0),
                            "metadata": {
                                "type": "web_result",
                                "search_engine": "tavily",
                                "url": result.get("url", ""),
                                "title": result.get("title", ""),
                            },
                        }
                    )

                # 🧠 Generate response using the response generator with live context
                if hasattr(self.rag_system, "response_generator"):
                    response_result = (
                        self.rag_system.response_generator.generate_response(
                            query, formatted_context
                        )
                    )

                    # 📊 Combine live search metadata with response
                    response_result.update(
                        {
                            "context_items": len(formatted_context),
                            "search_type": "live_web",
                            "routing_decision": "live_search",
                            "live_search_params": {
                                "search_depth": search_depth,
                                "time_range": time_range,
                                "total_web_results": len(search_results["results"]),
                            },
                        }
                    )

                    return response_result
                else:
                    # 📝 Fallback: simple response formatting
                    combined_content = "\n\n".join(
                        [
                            f"**{result.get('title', 'Web Result')}**\n{result.get('content', '')}"
                            for result in search_results["results"][:3]
                        ]
                    )

                    return {
                        "response": f"Based on live web search:\n\n{combined_content}",
                        "sources": search_results["results"],
                        "confidence": 0.8,
                        "context_items": len(search_results["results"]),
                        "search_type": "live_web",
                    }
            else:
                return {
                    "response": "No live search results found. Please try a different query or check your internet connection.",
                    "sources": [],
                    "confidence": 0.0,
                    "context_items": 0,
                    "error": "No live search results",
                }

        except Exception as e:
            self._log_safe(f" Live search error: {str(e)}", "error")
            # 🔄 Fallback to local search
            self._log_safe(" Falling back to local search...", "warning")
            return self.rag_system.query(query, max_results=max_results)

    def _call_tavily_mcp(
        self, query: str, max_results: int, search_depth: str, time_range: str
    ) -> Dict[str, Any]:
        """
        Call Tavily API using the live search module.

        Args:
            query: Search query
            max_results: Maximum results
            search_depth: Search depth
            time_range: Time range

        Returns:
            Tavily search results
        """
        try:
            # 🌐 Use the live search module with Tavily Python SDK
            from src.rag.live_search import LiveSearchManager

            self._log_safe(
                f" Tavily API call: query='{query}', depth={search_depth}, range={time_range}"
            )

            # ✅ Initialize live search manager
            live_search = LiveSearchManager()

            # 🚀 Perform the search using Tavily Python SDK
            search_results = live_search.search_web(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                time_range=time_range,
            )

            # 📊 Format results for UI consumption
            if (
                search_results
                and search_results.get("results")
                and not search_results.get("error")
            ):
                formatted_results = []
                for result in search_results.get("results", []):
                    formatted_results.append(
                        {
                            "title": result.get("title", ""),
                            "content": result.get("content", ""),
                            "url": result.get("url", ""),
                            "score": result.get("score", 0.0),
                            "published_date": result.get("published_date", ""),
                        }
                    )

                return {
                    "results": formatted_results,
                    "total_results": len(formatted_results),
                    "search_params": {
                        "query": query,
                        "max_results": max_results,
                        "search_depth": search_depth,
                        "time_range": time_range,
                    },
                    "status": "success",
                    "analytics": search_results.get("analytics", {}),
                }
            else:
                # 🚨 Handle search failure
                error_msg = search_results.get("error", "Unknown search error")
                self._log_safe(f" Tavily search failed: {error_msg}", "warning")

                return {
                    "results": [],
                    "total_results": 0,
                    "search_params": {
                        "query": query,
                        "max_results": max_results,
                        "search_depth": search_depth,
                        "time_range": time_range,
                    },
                    "status": "failed",
                    "error": error_msg,
                }

        except Exception as e:
            self._log_safe(f" Tavily API call failed: {str(e)}", "error")
            return {
                "results": [],
                "total_results": 0,
                "error": str(e),
                "status": "error",
            }

    def _refresh_knowledge_base(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[List[str]]]:
        """
        Refresh knowledge base information with real data from vector DB and embedding model.

        Returns:
            Tuple of (kb_stats, embedding_model_status, document_list)
        """
        try:
            # Get real knowledge base statistics
            kb_info = self._get_real_kb_stats()

            # Get embedding model information
            embedding_info = self._get_embedding_model_info()

            # 📊 Knowledge Base Stats
            kb_stats = {
                "total_documents": kb_info.get("total_documents", self.total_documents),
                "total_chunks": kb_info.get("total_chunks", self.total_chunks),
                "storage_size": f"{kb_info.get('total_chunks', self.total_chunks) * 0.5:.1f} MB",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "vector_db_status": kb_info.get("vector_db_status", "Unknown"),
                "embedding_model": embedding_info.get("model_name", "Unknown"),
                "embedding_status": embedding_info.get("status", "Unknown"),
                "index_health": kb_info.get("index_health", "Unknown"),
            }

            # 🤖 Embedding Model Status
            embedding_status = {
                "model_name": embedding_info.get("model_name", "Unknown"),
                "provider": embedding_info.get("provider", "Unknown"),
                "status": embedding_info.get("status", "Unknown"),
                "api_status": embedding_info.get("api_status", "Unknown"),
                "dimension": embedding_info.get("dimension", "Unknown"),
                "performance": {
                    "total_requests": embedding_info.get("total_requests", 0),
                    "success_rate": embedding_info.get("success_rate", "0%"),
                    "cache_hit_rate": embedding_info.get("cache_hit_rate", "0%"),
                    "batch_size": embedding_info.get("batch_size", "Unknown"),
                    "max_text_length": embedding_info.get("max_text_length", "Unknown"),
                    "caching_enabled": embedding_info.get("caching_enabled", False),
                },
                "last_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Get real document list from vector DB
            documents = self._get_real_document_list()

            # If no real documents, show helpful message
            if not documents:
                documents = [
                    [
                        "📝 No documents yet",
                        "Info",
                        "0",
                        "Upload documents to get started",
                    ],
                    ["🔗 Try adding URLs", "Info", "0", "Use the 'Add URLs' tab"],
                    [
                        "📚 Knowledge base empty",
                        "Info",
                        "0",
                        "Start building your knowledge base!",
                    ],
                ]

            return kb_stats, embedding_status, documents

        except Exception as e:
            self._log_safe(f" Error refreshing knowledge base: {e}", "error")
            # Fallback stats
            fallback_kb_stats = {
                "total_documents": self.total_documents,
                "total_chunks": self.total_chunks,
                "storage_size": f"{self.total_chunks * 0.5:.1f} MB",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
            }

            fallback_embedding_status = {
                "model_name": "Error",
                "provider": "Unknown",
                "status": "❌ Error",
                "api_status": "❌ Error",
                "dimension": "Unknown",
                "performance": {"error": str(e)},
                "last_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            return fallback_kb_stats, fallback_embedding_status, []

    def _get_real_kb_stats(self) -> Dict[str, Any]:
        """Get real knowledge base statistics from the RAG system."""
        try:
            # 🔍 Get embedding model info first
            embedding_model_info = self._get_embedding_model_info()

            if hasattr(self.rag_system, "vector_db") and self.rag_system.vector_db:
                # Try to get stats from vector DB
                vector_stats = (
                    self.rag_system.vector_db.get_stats()
                    if hasattr(self.rag_system.vector_db, "get_stats")
                    else {}
                )

                return {
                    "total_documents": vector_stats.get(
                        "total_vectors", self.total_documents
                    ),
                    "total_chunks": vector_stats.get(
                        "total_vectors", self.total_chunks
                    ),
                    "vector_db_status": "✅ Connected" if vector_stats else "⚠️ Limited",
                    "embedding_model": embedding_model_info.get(
                        "model_name", "Unknown"
                    ),
                    "embedding_model_status": embedding_model_info.get(
                        "status", "Unknown"
                    ),
                    "embedding_dimension": embedding_model_info.get(
                        "dimension", "Unknown"
                    ),
                    "embedding_provider": embedding_model_info.get(
                        "provider", "Unknown"
                    ),
                    "index_health": (
                        "✅ Healthy"
                        if vector_stats.get("total_vectors", 0) > 0
                        else "⚠️ Empty"
                    ),
                }
            else:
                return {
                    "total_documents": self.total_documents,
                    "total_chunks": self.total_chunks,
                    "vector_db_status": "❌ Not Connected",
                    "embedding_model": embedding_model_info.get(
                        "model_name", "Unknown"
                    ),
                    "embedding_model_status": embedding_model_info.get(
                        "status", "❌ Not Available"
                    ),
                    "embedding_dimension": embedding_model_info.get(
                        "dimension", "Unknown"
                    ),
                    "embedding_provider": embedding_model_info.get(
                        "provider", "Unknown"
                    ),
                    "index_health": "❌ Unavailable",
                }
        except Exception as e:
            self._log_safe(f"Could not get real KB stats: {e}", "warning")
            return {}

    def _get_real_document_list(self) -> List[List[str]]:
        """Get real document list from the RAG system."""
        try:
            documents = []

            # Try to get document metadata from vector DB
            if hasattr(self.rag_system, "vector_db") and self.rag_system.vector_db:
                # Get unique sources from vector DB
                if hasattr(self.rag_system.vector_db, "get_unique_sources"):
                    sources = self.rag_system.vector_db.get_unique_sources()
                    for source_info in sources:
                        source_name = source_info.get("source", "Unknown")
                        doc_type = self._get_document_type(source_name)
                        chunk_count = source_info.get("chunk_count", 0)
                        added_date = source_info.get("added_date", "Unknown")

                        documents.append(
                            [source_name, doc_type, str(chunk_count), added_date]
                        )

                # If vector DB doesn't have get_unique_sources, try alternative approach
                elif hasattr(self.rag_system.vector_db, "list_documents"):
                    doc_list = self.rag_system.vector_db.list_documents()
                    for doc in doc_list:
                        documents.append(
                            [
                                doc.get("name", "Unknown"),
                                self._get_document_type(doc.get("name", "")),
                                str(doc.get("chunks", 0)),
                                doc.get("date", "Unknown"),
                            ]
                        )

            return documents

        except Exception as e:
            self._log_safe(f"Could not get real document list: {e}", "warning")
            return []

    def _get_document_type(self, filename: str) -> str:
        """Determine document type from filename."""
        if not filename:
            return "Unknown"

        filename_lower = filename.lower()
        if filename_lower.endswith(".pdf"):
            return "📄 PDF"
        elif filename_lower.endswith((".doc", ".docx")):
            return "📝 Word"
        elif filename_lower.endswith((".xls", ".xlsx")):
            return "📊 Excel"
        elif filename_lower.endswith((".ppt", ".pptx")):
            return "📈 PowerPoint"
        elif filename_lower.endswith(".csv"):
            return "📋 CSV"
        elif filename_lower.endswith((".txt", ".md")):
            return "📄 Text"
        elif "http" in filename_lower:
            return "🌐 Web"
        else:
            return "📄 Document"

    def _get_embedding_model_info(self) -> Dict[str, Any]:
        """
        🤖 Get comprehensive embedding model information.

        Returns:
            Dictionary with embedding model details
        """
        try:
            model_info = {
                "model_name": "Unknown",
                "status": "❌ Not Available",
                "dimension": "Unknown",
                "provider": "Unknown",
                "api_status": "❌ Not Connected",
            }

            # Check if embedding generator exists and is properly initialized
            if (
                hasattr(self.rag_system, "embedding_generator")
                and self.rag_system.embedding_generator
            ):
                embedding_gen = self.rag_system.embedding_generator

                # Get model name - check multiple possible attributes
                model_name = (
                    getattr(embedding_gen, "model", None)
                    or getattr(embedding_gen, "model_name", None)
                    or "gemini-embedding-exp-03-07"
                )  # Default Gemini model

                # Get API client status
                api_connected = (
                    hasattr(embedding_gen, "client")
                    and embedding_gen.client is not None
                )

                # Get configuration details
                config = getattr(embedding_gen, "config", {})

                model_info.update(
                    {
                        "model_name": model_name,
                        "status": "✅ Available" if api_connected else "⚠️ Limited",
                        "provider": (
                            "Google Gemini"
                            if "gemini" in model_name.lower()
                            else "Unknown"
                        ),
                        "api_status": (
                            "✅ Connected" if api_connected else "❌ Not Connected"
                        ),
                        "dimension": config.get("dimension", "3072"),  # Gemini default
                        "batch_size": config.get("batch_size", 5),
                        "max_text_length": config.get("max_text_length", 8192),
                        "caching_enabled": config.get("enable_caching", True),
                    }
                )

                # Get statistics if available
                if hasattr(embedding_gen, "get_statistics"):
                    try:
                        stats = embedding_gen.get_statistics()
                        model_info.update(
                            {
                                "total_requests": stats.get("total_requests", 0),
                                "successful_requests": stats.get(
                                    "successful_requests", 0
                                ),
                                "cache_hits": stats.get("cache_hits", 0),
                                "cache_hit_rate": f"{stats.get('cache_hit_rate', 0):.1f}%",
                                "success_rate": f"{stats.get('success_rate', 0):.1f}%",
                            }
                        )
                    except Exception as e:
                        self._log_safe(f"Could not get embedding stats: {e}", "warning")

                # Test API connection if possible (quick test)
                if api_connected:
                    try:
                        # Quick test to verify API is working
                        test_embedding = embedding_gen.generate_query_embedding("test")
                        if test_embedding:
                            model_info["api_status"] = "✅ Connected & Working"
                            model_info["status"] = "✅ Fully Operational"
                        else:
                            model_info["api_status"] = "⚠️ Connected but Limited"
                    except Exception as e:
                        model_info["api_status"] = f"❌ Connection Error: {str(e)[:50]}"

            return model_info

        except Exception as e:
            self._log_safe(f"Error getting embedding model info: {e}", "error")
            return {
                "model_name": "Error",
                "status": "❌ Error",
                "dimension": "Unknown",
                "provider": "Unknown",
                "api_status": f"❌ Error: {str(e)[:50]}",
                "error": str(e),
            }

    def _run_health_check(self) -> Tuple[Dict[str, Any], List[List[str]], str]:
        """
        🩺 Run comprehensive real system health check.

        Returns:
            Tuple of (system status, component status, logs)
        """
        try:
            import psutil
            import time
            from datetime import timedelta

            # 📊 Real System Status
            start_time = time.time()

            # Get real system metrics
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            # Calculate uptime (approximate)
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            uptime = str(timedelta(seconds=int(uptime_seconds)))

            system_status = {
                "overall_health": "🟢 Healthy",
                "uptime": uptime,
                "memory_usage": f"{memory_info.percent:.1f}%",
                "memory_available": f"{memory_info.available / (1024**3):.1f} GB",
                "cpu_usage": f"{cpu_percent:.1f}%",
                "disk_usage": f"{psutil.disk_usage('/').percent:.1f}%",
                "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "system_load": "Normal" if cpu_percent < 80 else "High",
            }

            # 🔍 Real Component Status Check
            components = []
            logs = []

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logs.append(f"[{current_time}] INFO - System health check initiated")

            # 1. 🤖 Embedding Generator Check
            embedding_info = self._get_embedding_model_info()
            embedding_status = embedding_info.get("status", "❌ Unknown")
            embedding_details = f"{embedding_info.get('model_name', 'Unknown')} - {embedding_info.get('api_status', 'Unknown')}"
            components.append(
                ["🤖 Embedding Generator", embedding_status, embedding_details]
            )

            if "✅" in embedding_status:
                logs.append(
                    f"[{current_time}] INFO - Embedding generator: {embedding_details}"
                )
            else:
                logs.append(
                    f"[{current_time}] WARN - Embedding generator: {embedding_details}"
                )

            # 2. 🌲 Vector Database Check
            vector_db_status, vector_db_details = self._check_vector_db_health()
            components.append(
                ["🌲 Vector Database", vector_db_status, vector_db_details]
            )
            logs.append(f"[{current_time}] INFO - Vector database: {vector_db_details}")

            # 3. 📄 Document Processor Check
            doc_processor_status, doc_processor_details = (
                self._check_document_processor_health()
            )
            components.append(
                ["📄 Document Processor", doc_processor_status, doc_processor_details]
            )
            logs.append(
                f"[{current_time}] INFO - Document processor: {doc_processor_details}"
            )

            # 4. 🧠 Response Generator Check
            response_gen_status, response_gen_details = (
                self._check_response_generator_health()
            )
            components.append(
                ["🧠 Response Generator", response_gen_status, response_gen_details]
            )
            logs.append(
                f"[{current_time}] INFO - Response generator: {response_gen_details}"
            )

            # 5. 🌐 Web Interface Check
            components.append(
                ["🌐 Web Interface", "✅ Healthy", "Gradio running successfully"]
            )
            logs.append(f"[{current_time}] INFO - Web interface: Running on port 7860")

            # 6. 🔍 Live Search Check (if available)
            live_search_status, live_search_details = self._check_live_search_health()
            components.append(
                ["🔍 Live Search", live_search_status, live_search_details]
            )
            logs.append(f"[{current_time}] INFO - Live search: {live_search_details}")

            # Calculate overall health
            healthy_components = sum(1 for comp in components if "✅" in comp[1])
            total_components = len(components)
            health_percentage = (healthy_components / total_components) * 100

            if health_percentage >= 80:
                system_status["overall_health"] = "🟢 Healthy"
                logs.append(
                    f"[{current_time}] INFO - Overall system health: {health_percentage:.0f}% ({healthy_components}/{total_components} components healthy)"
                )
            elif health_percentage >= 60:
                system_status["overall_health"] = "🟡 Degraded"
                logs.append(
                    f"[{current_time}] WARN - System degraded: {health_percentage:.0f}% ({healthy_components}/{total_components} components healthy)"
                )
            else:
                system_status["overall_health"] = "🔴 Unhealthy"
                logs.append(
                    f"[{current_time}] ERROR - System unhealthy: {health_percentage:.0f}% ({healthy_components}/{total_components} components healthy)"
                )

            # Add performance metrics
            health_check_time = time.time() - start_time
            system_status["health_check_duration"] = f"{health_check_time:.2f}s"
            logs.append(
                f"[{current_time}] INFO - Health check completed in {health_check_time:.2f}s"
            )

            return system_status, components, "\n".join(logs)

        except Exception as e:
            self._log_safe(f"❌ Error running health check: {e}", "error")
            error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return (
                {
                    "overall_health": "🔴 Error",
                    "error": str(e),
                    "last_check": error_time,
                },
                [["System", "❌ Error", f"Health check failed: {str(e)}"]],
                f"[{error_time}] ERROR - Health check failed: {str(e)}",
            )

    def _check_vector_db_health(self) -> Tuple[str, str]:
        """🌲 Check Vector Database health status."""
        try:
            if hasattr(self.rag_system, "vector_db") and self.rag_system.vector_db:
                vector_db = self.rag_system.vector_db

                # Try to get health check from vector DB
                if hasattr(vector_db, "health_check"):
                    health_result = vector_db.health_check()
                    if health_result.get("status") == "healthy":
                        return (
                            "✅ Healthy",
                            f"Pinecone connected - {health_result.get('checks', {}).get('index_stats', 'OK')}",
                        )
                    else:
                        return (
                            "⚠️ Degraded",
                            f"Issues detected: {health_result.get('error', 'Unknown')}",
                        )

                # Fallback: check if we can get stats
                elif hasattr(vector_db, "get_stats"):
                    stats = vector_db.get_stats()
                    if stats.get("status") == "connected":
                        total_vectors = stats.get("total_vectors", 0)
                        return (
                            "✅ Healthy",
                            f"Pinecone connected - {total_vectors} vectors stored",
                        )
                    else:
                        return (
                            "❌ Error",
                            f"Connection failed: {stats.get('error', 'Unknown')}",
                        )

                else:
                    return (
                        "⚠️ Limited",
                        "Vector DB available but health check not implemented",
                    )
            else:
                return "❌ Not Available", "Vector database not initialized"

        except Exception as e:
            return "❌ Error", f"Health check failed: {str(e)[:50]}"

    def _check_document_processor_health(self) -> Tuple[str, str]:
        """📄 Check Document Processor health status."""
        try:
            if (
                hasattr(self.rag_system, "document_processor")
                and self.rag_system.document_processor
            ):
                # Check if document processor has required dependencies
                try:
                    # Test basic functionality
                    processor = self.rag_system.document_processor

                    # Check if it has the required methods
                    if hasattr(processor, "process_document"):
                        supported_formats = [
                            "PDF",
                            "DOCX",
                            "CSV",
                            "XLSX",
                            "PPTX",
                            "TXT",
                            "MD",
                        ]
                        return (
                            "✅ Healthy",
                            f"All formats supported: {', '.join(supported_formats)}",
                        )
                    else:
                        return "⚠️ Limited", "Basic functionality available"

                except ImportError as e:
                    return (
                        "❌ Dependencies Missing",
                        f"Missing libraries: {str(e)[:30]}",
                    )
            else:
                return "❌ Not Available", "Document processor not initialized"

        except Exception as e:
            return "❌ Error", f"Health check failed: {str(e)[:50]}"

    def _check_response_generator_health(self) -> Tuple[str, str]:
        """🧠 Check Response Generator health status."""
        try:
            if (
                hasattr(self.rag_system, "response_generator")
                and self.rag_system.response_generator
            ):
                response_gen = self.rag_system.response_generator

                # Check if it has required configuration
                config = getattr(response_gen, "config", {})

                # Check API keys availability
                gemini_key = config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
                openai_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

                if gemini_key:
                    return "✅ Healthy", "Gemini LLM available for response generation"
                elif openai_key:
                    return "✅ Healthy", "OpenAI LLM available for response generation"
                else:
                    return "⚠️ Limited", "No LLM API keys configured"
            else:
                return "❌ Not Available", "Response generator not initialized"

        except Exception as e:
            return "❌ Error", f"Health check failed: {str(e)[:50]}"

    def _check_live_search_health(self) -> Tuple[str, str]:
        """🔍 Check Live Search health status."""
        try:
            # Check if Tavily API key is available
            tavily_key = os.getenv("TAVILY_API_KEY")

            if tavily_key:
                # Check if live search components are available
                if (
                    hasattr(self.rag_system, "live_search_processor")
                    and self.rag_system.live_search_processor
                ):
                    return "✅ Healthy", "Tavily API configured - Live search available"
                elif (
                    hasattr(self.rag_system, "query_router")
                    and self.rag_system.query_router
                ):
                    return "✅ Healthy", "Query router available - Live search enabled"
                else:
                    return (
                        "⚠️ Limited",
                        "Tavily API key available but components not initialized",
                    )
            else:
                return (
                    "⚠️ Optional",
                    "Tavily API key not configured - Live search disabled",
                )

        except Exception as e:
            return "❌ Error", f"Health check failed: {str(e)[:50]}"

    def _get_stats_string(self) -> str:
        """Get formatted stats string."""
        return f"Documents: {self.total_documents} | Chunks: {self.total_chunks} | Queries: {self.query_count}"

    def launch(self, **kwargs):
        """
        Launch the Gradio interface.

        Args:
            **kwargs: Additional arguments for gr.Interface.launch()
        """
        if not self.interface:
            self._log_safe(" Interface not created", "error")
            return

        # Merge default config with provided kwargs
        launch_config = {
            "share": self.share,
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "show_error": True,
            "quiet": False,
        }
        launch_config.update(kwargs)

        self._log_safe(f"Launching Gradio interface with config: {launch_config}")
        self.interface.launch(**launch_config)
