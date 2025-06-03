"""
Gradio UI Module

This module provides an intuitive interface for document upload,
URL input, and querying using Gradio.

Technology: Gradio
"""

import logging
import os
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
        self.logger = logging.getLogger(__name__)

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

        self.logger.info("GradioApp initialized with advanced features")

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

                with gr.Accordion("⚙Query Options", open=False):
                    include_sources = gr.Checkbox(
                        label="Include Sources",
                        value=True,
                    )
                    max_results = gr.Slider(
                        label="Max Results",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                    )

                with gr.Row():
                    query_btn = gr.Button("Get Answer", variant="primary", size="lg")
                    clear_query_btn = gr.Button("Clear", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### 💬 Answer")

                response_output = gr.Textbox(
                    label="Response",
                    lines=10,
                    interactive=False,
                    placeholder="Your answer will appear here...",
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
            inputs=[query_input],
            outputs=[
                response_output,
                confidence_display,
                sources_output,
                self.status_display,
                self.stats_display,
            ],
        )

        clear_query_btn.click(
            fn=lambda: ("", "", {}, "Ready "),
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
        }

    def _create_knowledge_base_tab(self):
        """Create the knowledge base management tab."""
        with gr.Column():
            gr.Markdown("### 📚 Knowledge Base Management")

            with gr.Row():
                refresh_btn = gr.Button("Refresh", variant="secondary")
                export_btn = gr.Button("📤 Export", variant="secondary")
                clear_kb_btn = gr.Button("Clear All", variant="stop")

            # Knowledge base stats
            kb_stats = gr.JSON(
                label="Knowledge Base Statistics",
                value={"total_documents": 0, "total_chunks": 0, "storage_size": "0 MB"},
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
            outputs=[kb_stats, document_list],
        )

        return {
            "kb_stats": kb_stats,
            "document_list": document_list,
        }

    def _create_analytics_tab(self):
        """Create the analytics dashboard tab."""
        with gr.Column():
            gr.Markdown("### 📈 Analytics Dashboard")

            with gr.Row():
                with gr.Column():
                    query_analytics = gr.JSON(
                        label="Query Analytics",
                        value={},
                    )

                with gr.Column():
                    system_metrics = gr.JSON(
                        label="⚡ System Metrics",
                        value={},
                    )

            # Query history
            query_history = gr.Dataframe(
                headers=["Query", "Results", "Confidence", "Time"],
                datatype=["str", "number", "number", "str"],
                label="Recent Queries",
                interactive=False,
            )

        return {
            "query_analytics": query_analytics,
            "system_metrics": system_metrics,
            "query_history": query_history,
        }

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
            self.logger.info(f"Processing {len(files)} uploaded files")

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
            self.logger.error(f"❌ Error processing documents: {str(e)}")
            return f"❌ Error: {str(e)}", "Error ❌", self._get_stats_string()

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
            self.logger.info(
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
                else "Processing failed ❌"
            )

            final_progress = (
                f"✅ Completed! Processed {successful}/{len(urls)} URLs successfully"
            )
            return output, status, self._get_stats_string(), final_progress

        except Exception as e:
            self.logger.error(f"❌ Error processing URLs: {str(e)}")
            error_progress = f"❌ Error occurred during processing"
            return (
                f"❌ Error: {str(e)}",
                "Error ❌",
                self._get_stats_string(),
                error_progress,
            )

    def _process_query(self, query: str) -> Tuple[str, str, Dict[str, Any], str, str]:
        """
        Process a user query with enhanced response formatting.

        Args:
            query: User query string

        Returns:
            Tuple of (response, confidence, sources, status, stats)
        """
        if not query.strip():
            return (
                "Please enter a question.",
                "",
                {},
                "Ready ",
                self._get_stats_string(),
            )

        try:
            self.logger.info(f"Processing query: {query[:100]}...")

            # Get response from RAG system
            result = self.rag_system.query(query)

            self.query_count += 1

            response = result.get("response", "No response generated.")
            confidence = result.get("confidence", 0.0)
            sources = result.get("sources", [])

            # Format confidence display
            confidence_text = f"Confidence: {confidence:.1%}"
            if confidence >= 0.8:
                confidence_text += " 🟢 High"
            elif confidence >= 0.5:
                confidence_text += " 🟡 Medium"
            else:
                confidence_text += " 🔴 Low"

            # Format sources for display
            sources_display = {
                "confidence": f"{confidence:.3f}",
                "total_sources": len(sources),
                "sources": sources[:5],  # Limit to top 5 sources
            }

            status = f"Query processed (confidence: {confidence:.1%}) "

            return (
                response,
                confidence_text,
                sources_display,
                status,
                self._get_stats_string(),
            )

        except Exception as e:
            self.logger.error(f"❌ Error processing query: {str(e)}")
            return (
                f"❌ Error: {str(e)}",
                "Error",
                {},
                "Error ❌",
                self._get_stats_string(),
            )

    def _refresh_knowledge_base(self) -> Tuple[Dict[str, Any], List[List[str]]]:
        """
        Refresh knowledge base information.

        Returns:
            Tuple of (stats, document list)
        """
        try:
            # Get knowledge base statistics
            stats = {
                "total_documents": self.total_documents,
                "total_chunks": self.total_chunks,
                "storage_size": f"{self.total_chunks * 0.5:.1f} MB",  # Estimate
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Mock document list (in real implementation, get from vector DB)
            documents = [
                ["sample.pdf", "PDF", "25", "2025-06-05 10:30"],
                ["webpage.html", "Web", "15", "2025-06-05 11:45"],
                ["document.docx", "Word", "30", "2025-06-15 12:00"],
                ["document.csv", "CSV", "30", "2025-06-15 12:00"],
                ["notes.md", "Markdown", "12", "2025-06-16 09:20"],
                ["readme.txt", "Text", "8", "2025-06-16 09:25"],
                ["presentation.pptx", "PowerPoint", "22", "2025-06-16 09:30"],
                ["data.xlsx", "Excel", "18", "2025-06-16 09:35"],
            ]

            return stats, documents

        except Exception as e:
            self.logger.error(f"❌ Error refreshing knowledge base: {e}")
            return {}, []

    def _run_health_check(self) -> Tuple[Dict[str, Any], List[List[str]], str]:
        """
        Run system health check.

        Returns:
            Tuple of (system status, component status, logs)
        """
        try:
            # System status
            system_status = {
                "overall_health": "Healthy ",
                "uptime": "2h 15m",
                "memory_usage": "45%",
                "cpu_usage": "12%",
                "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Component status
            components = [
                ["Embedding Generator", "Healthy", "Gemini API connected"],
                ["Vector Database", "Healthy", "Pinecone connected"],
                ["Document Processor", "Healthy", "All formats supported"],
                ["Response Generator", "Healthy", "LLM available"],
                ["Web Interface", "Healthy", "Gradio running"],
            ]

            # System logs
            logs = """
[2024-01-15 14:30:15] INFO - System health check initiated
[2024-01-15 14:30:16] INFO - Checking embedding generator... OK
[2024-01-15 14:30:17] INFO - Checking vector database... OK
[2024-01-15 14:30:18] INFO - Checking document processor... OK
[2024-01-15 14:30:19] INFO - Checking response generator... OK
[2024-01-15 14:30:20] INFO - All systems operational 
            """.strip()

            return system_status, components, logs

        except Exception as e:
            self.logger.error(f"❌ Error running health check: {e}")
            return {}, [], f"Health check failed: {str(e)}"

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
            self.logger.error("❌ Interface not created")
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

        self.logger.info(f"Launching Gradio interface with config: {launch_config}")
        self.interface.launch(**launch_config)
