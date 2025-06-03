"""
Ingestion Pipeline Module

This module orchestrates the complete document ingestion process,
integrating all components for a seamless workflow.

Components: DocumentProcessor, URLProcessor, TextExtractor, EmbeddingGenerator, VectorDB
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import asyncio
from datetime import datetime

from .document_processor import DocumentProcessor
from .url_processor import URLProcessor
from ingestion.text_extractor import TextExtractor
from embedding.embedding_generator import EmbeddingGenerator
from storage.vector_db import VectorDB
from utils.config_manager import ConfigManager
from utils.error_handler import error_handler, ErrorType, RAGError


class IngestionPipeline:
    """
    Complete ingestion pipeline that orchestrates document processing, text extraction,
    embedding generation, and vector storage.

    Features:
    - End-to-end document ingestion 
    - URL content processing 
    - Batch processing capabilities 
    - Progress tracking and statistics 
    - Error handling and recovery 
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ingestion pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config

        # Initialize statistics
        self.stats = {
            "documents_processed": 0,
            "urls_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "vectors_stored": 0,
            "errors_encountered": 0,
            "start_time": None,
            "end_time": None,
        }

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # 📄 Document processor
            doc_config = self.config.get("document_processing", {})
            self.document_processor = DocumentProcessor(doc_config)

            # URL processor
            url_config = self.config.get("url_processing", {})
            self.url_processor = URLProcessor(url_config)

            # Text extractor
            text_config = self.config.get("document_processing", {})
            self.text_extractor = TextExtractor(text_config)

            # 🔮 Embedding generator
            embedding_config = self.config.get("embedding", {})
            embedding_config["api_key"] = self.config.get("api_keys", {}).get(
                "gemini_api_key"
            )
            self.embedding_generator = EmbeddingGenerator(embedding_config)

            # Vector database
            vector_config = self.config.get("vector_db", {})
            vector_config["api_key"] = self.config.get("api_keys", {}).get(
                "pinecone_api_key"
            )
            self.vector_db = VectorDB(vector_config)

            self.logger.info("All pipeline components initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize pipeline components: {str(e)}")
            raise RAGError(f"Pipeline initialization failed: {str(e)}")

    @error_handler(ErrorType.DOCUMENT_PROCESSING)
    def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple documents through the complete pipeline.

        Args:
            file_paths: List of document file paths

        Returns:
            Processing results and statistics
        """
        self.logger.info(
            f"Starting document processing pipeline for {len(file_paths)} files"
        )
        self.stats["start_time"] = datetime.now()

        all_results = []

        for i, file_path in enumerate(file_paths):
            try:
                self.logger.info(
                    f"📄 Processing document {i+1}/{len(file_paths)}: {file_path}"
                )

                # 📄 Step 1: Process document
                doc_result = self.document_processor.process_document(file_path)
                self.stats["documents_processed"] += 1

                # Step 2: Extract and chunk text
                text_chunks = self.text_extractor.process_text(
                    doc_result["content"], doc_result["metadata"]
                )
                self.stats["chunks_created"] += len(text_chunks)

                # 🔮 Step 3: Generate embeddings
                embedded_chunks = self.embedding_generator.generate_embeddings(
                    text_chunks
                )
                valid_embeddings = [
                    chunk for chunk in embedded_chunks if chunk.get("embedding")
                ]
                self.stats["embeddings_generated"] += len(valid_embeddings)

                # Step 4: Store in vector database
                if valid_embeddings:
                    storage_success = self.vector_db.store_embeddings(valid_embeddings)
                    if storage_success:
                        self.stats["vectors_stored"] += len(valid_embeddings)

                # Compile results
                result = {
                    "file_path": file_path,
                    "document_type": doc_result.get("document_type"),
                    "chunks_created": len(text_chunks),
                    "embeddings_generated": len(valid_embeddings),
                    "storage_success": storage_success if valid_embeddings else False,
                    "metadata": doc_result["metadata"],
                }

                all_results.append(result)
                self.logger.info(
                    f"Document processed: {len(text_chunks)} chunks, {len(valid_embeddings)} embeddings"
                )

            except Exception as e:
                self.stats["errors_encountered"] += 1
                self.logger.error(f"❌ Error processing {file_path}: {str(e)}")

                all_results.append(
                    {
                        "file_path": file_path,
                        "error": str(e),
                        "chunks_created": 0,
                        "embeddings_generated": 0,
                        "storage_success": False,
                    }
                )

        self.stats["end_time"] = datetime.now()

        return {
            "results": all_results,
            "statistics": self.get_statistics(),
            "success_rate": self._calculate_success_rate(all_results),
        }

    @error_handler(ErrorType.URL_PROCESSING)
    def process_urls(self, urls: List[str]) -> Dict[str, Any]:
        """
        Process multiple URLs through the complete pipeline.

        Args:
            urls: List of URLs to process

        Returns:
            Processing results and statistics
        """
        self.logger.info(f"Starting URL processing pipeline for {len(urls)} URLs")
        self.stats["start_time"] = datetime.now()

        all_results = []

        for i, url in enumerate(urls):
            try:
                self.logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")

                # Step 1: Process URL
                url_result = self.url_processor.process_url(url)
                if not url_result:
                    self.logger.warning(f"No content extracted from URL: {url}")
                    continue

                self.stats["urls_processed"] += 1

                # Step 2: Extract and chunk text
                text_chunks = self.text_extractor.process_text(
                    url_result["content"], url_result["metadata"]
                )
                self.stats["chunks_created"] += len(text_chunks)

                # 🔮 Step 3: Generate embeddings
                embedded_chunks = self.embedding_generator.generate_embeddings(
                    text_chunks
                )
                valid_embeddings = [
                    chunk for chunk in embedded_chunks if chunk.get("embedding")
                ]
                self.stats["embeddings_generated"] += len(valid_embeddings)

                # Step 4: Store in vector database
                storage_success = False
                if valid_embeddings:
                    storage_success = self.vector_db.store_embeddings(valid_embeddings)
                    if storage_success:
                        self.stats["vectors_stored"] += len(valid_embeddings)

                # Process linked documents if any
                linked_results = []
                for linked_doc in url_result.get("linked_documents", []):
                    if linked_doc.get("content"):
                        linked_chunks = self.text_extractor.process_text(
                            linked_doc["content"], linked_doc["metadata"]
                        )
                        linked_embedded = self.embedding_generator.generate_embeddings(
                            linked_chunks
                        )
                        linked_valid = [
                            chunk for chunk in linked_embedded if chunk.get("embedding")
                        ]

                        if linked_valid:
                            self.vector_db.store_embeddings(linked_valid)
                            linked_results.append(
                                {
                                    "url": linked_doc["source"],
                                    "chunks": len(linked_chunks),
                                    "embeddings": len(linked_valid),
                                }
                            )

                # Compile results
                result = {
                    "url": url,
                    "chunks_created": len(text_chunks),
                    "embeddings_generated": len(valid_embeddings),
                    "storage_success": storage_success,
                    "linked_documents": linked_results,
                    "metadata": url_result["metadata"],
                }

                all_results.append(result)
                self.logger.info(
                    f"URL processed: {len(text_chunks)} chunks, {len(valid_embeddings)} embeddings"
                )

            except Exception as e:
                self.stats["errors_encountered"] += 1
                self.logger.error(f"❌ Error processing {url}: {str(e)}")

                all_results.append(
                    {
                        "url": url,
                        "error": str(e),
                        "chunks_created": 0,
                        "embeddings_generated": 0,
                        "storage_success": False,
                    }
                )

        self.stats["end_time"] = datetime.now()

        return {
            "results": all_results,
            "statistics": self.get_statistics(),
            "success_rate": self._calculate_success_rate(all_results),
        }

    def process_mixed_sources(
        self, file_paths: Optional[List[str]] = None, urls: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process both documents and URLs in a single pipeline run.

        Args:
            file_paths: Optional list of document file paths
            urls: Optional list of URLs

        Returns:
            Combined processing results
        """
        self.logger.info("Starting mixed source processing pipeline")

        results = {
            "document_results": [],
            "url_results": [],
            "combined_statistics": {},
            "overall_success_rate": 0.0,
        }

        # 📄 Process documents
        if file_paths:
            doc_results = self.process_documents(file_paths)
            results["document_results"] = doc_results["results"]

        # Process URLs
        if urls:
            url_results = self.process_urls(urls)
            results["url_results"] = url_results["results"]

        # Combine statistics
        results["combined_statistics"] = self.get_statistics()

        # 🎯 Calculate overall success rate
        all_items = results["document_results"] + results["url_results"]
        results["overall_success_rate"] = self._calculate_success_rate(all_items)

        return results

    def _calculate_success_rate(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate success rate from results.

        Args:
            results: List of processing results

        Returns:
            Success rate as percentage
        """
        if not results:
            return 0.0

        successful = sum(
            1 for result in results if result.get("storage_success", False)
        )
        return (successful / len(results)) * 100

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline statistics.

        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()

        if stats["start_time"] and stats["end_time"]:
            runtime = stats["end_time"] - stats["start_time"]
            stats["runtime_seconds"] = runtime.total_seconds()
            stats["processing_rate"] = (
                stats["documents_processed"] + stats["urls_processed"]
            ) / max(1, runtime.total_seconds())

        # 🔮 Add component statistics
        stats["embedding_stats"] = self.embedding_generator.get_statistics()
        stats["vector_db_stats"] = self.vector_db.get_index_stats()
        stats["url_processor_stats"] = self.url_processor.get_statistics()

        return stats

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on all components.

        Returns:
            Health check results
        """
        health = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        try:
            # 🔮 Check embedding generator
            if self.embedding_generator.client:
                health["components"]["embedding_generator"] = "Ready"
            else:
                health["components"]["embedding_generator"] = "❌ Not configured"
                health["overall_status"] = "degraded"

            # Check vector database
            vector_health = self.vector_db.health_check()
            health["components"]["vector_database"] = vector_health["status"]
            if vector_health["status"] != "healthy":
                health["overall_status"] = "degraded"

            # Add component details
            health["details"] = {
                "vector_db_health": vector_health,
                "embedding_stats": self.embedding_generator.get_statistics(),
                "pipeline_stats": self.get_statistics(),
            }

        except Exception as e:
            health["overall_status"] = "unhealthy"
            health["error"] = str(e)

        return health

    def reset_statistics(self):
        """Reset pipeline statistics."""
        self.stats = {
            "documents_processed": 0,
            "urls_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "vectors_stored": 0,
            "errors_encountered": 0,
            "start_time": None,
            "end_time": None,
        }

        # Reset component statistics
        self.embedding_generator.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_tokens_processed": 0,
            "start_time": datetime.now(),
        }

        self.vector_db.reset_stats()
        self.url_processor.reset()

        self.logger.info("All pipeline statistics reset")


# Convenience function for quick pipeline usage
def create_pipeline(config_path: Optional[str] = None) -> IngestionPipeline:
    """
    Create and return a configured ingestion pipeline.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Configured IngestionPipeline instance
    """
    return IngestionPipeline(config_path)


# 📄 Example usage functions
def process_documents_simple(
    file_paths: List[str], config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    📄 Simple function to process documents with default configuration.

    Args:
        file_paths: List of document file paths
        config_path: Optional configuration file path

    Returns:
        Processing results
    """
    pipeline = create_pipeline(config_path)
    return pipeline.process_documents(file_paths)


def process_urls_simple(
    urls: List[str], config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simple function to process URLs with default configuration.

    Args:
        urls: List of URLs to process
        config_path: Optional configuration file path

    Returns:
        Processing results
    """
    pipeline = create_pipeline(config_path)
    return pipeline.process_urls(urls)
