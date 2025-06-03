"""
Vector Database Module

This module is responsible for storing and indexing vector embeddings
for efficient retrieval using Pinecone with complete functionality.

Technology: Pinecone
"""

import logging
import os
import time
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Import Pinecone and related libraries
try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
except ImportError as e:
    logging.warning(f"Pinecone library not installed: {e}")

from utils.error_handler import VectorStorageError, error_handler, ErrorType


class VectorDB:
    """
    Stores and indexes vector embeddings for efficient retrieval using Pinecone with full functionality.

    Features:
    - Complete Pinecone integration
    - Index management (create, update, delete)
    - Batch upsert operations with optimization
    - Advanced similarity search with metadata filtering
    - Statistics and monitoring
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the VectorDB with configuration.

        Args:
            config: Configuration dictionary with Pinecone parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.api_key = self.config.get("api_key", os.environ.get("PINECONE_API_KEY"))
        self.environment = self.config.get("environment", "us-west1-gcp")
        self.index_name = self.config.get("index_name", "rag-ai-index")
        self.dimension = self.config.get(
            "dimension", 3072
        )  # ✅ Fixed: Match Gemini embedding dimension
        self.metric = self.config.get("metric", "cosine")
        self.batch_size = self.config.get("batch_size", 100)

        # Performance settings
        self.max_metadata_size = self.config.get(
            "max_metadata_size", 40960
        )  # 40KB limit
        self.upsert_timeout = self.config.get("upsert_timeout", 60)
        self.query_timeout = self.config.get("query_timeout", 30)

        # Statistics tracking
        self.stats = {
            "vectors_stored": 0,
            "vectors_queried": 0,
            "vectors_deleted": 0,
            "batch_operations": 0,
            "failed_operations": 0,
            "start_time": datetime.now(),
        }

        # Initialize Pinecone client
        self.pc = None
        self.index = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Pinecone client and index with validation."""
        if not self.api_key:
            self.logger.warning(
                "No Pinecone API key provided. Vector storage will not be available."
            )
            return

        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)

            # Check if index exists, create if not
            self._ensure_index_exists()

            # Connect to index
            self.index = self.pc.Index(self.index_name)

            # Test connection
            self._test_connection()

            self.logger.info(
                f"Pinecone client initialized successfully with index: {self.index_name}"
            )

        except Exception as e:
            self.logger.error(f" Failed to initialize Pinecone client: {str(e)}")
            self.pc = None
            self.index = None

    def _ensure_index_exists(self):
        """Ensure the Pinecone index exists, create if necessary."""
        try:
            # List existing indexes
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                self.logger.info(f"Creating new Pinecone index: {self.index_name}")

                # Create index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud="aws", region=self.environment),
                )

                # Wait for index to be ready
                self._wait_for_index_ready()

                self.logger.info(f"Index {self.index_name} created successfully")
            else:
                self.logger.info(f"Index {self.index_name} already exists")

        except Exception as e:
            raise VectorStorageError(f"Failed to ensure index exists: {str(e)}")

    def _wait_for_index_ready(self, max_wait_time: int = 300):
        """Wait for index to be ready for operations."""
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                index_stats = self.pc.describe_index(self.index_name)
                if index_stats.status.ready:
                    self.logger.info(f"Index {self.index_name} is ready")
                    return

                self.logger.info(f"Waiting for index to be ready...")
                time.sleep(10)

            except Exception as e:
                self.logger.warning(f"Error checking index status: {str(e)}")
                time.sleep(5)

        raise VectorStorageError(
            f"Index {self.index_name} not ready after {max_wait_time}s"
        )

    def _test_connection(self):
        """Test connection to Pinecone index."""
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            self.logger.info(f"Connection test successful. Index stats: {stats}")

        except Exception as e:
            raise VectorStorageError(f"Connection test failed: {str(e)}")

    @error_handler(ErrorType.VECTOR_STORAGE)
    def store_embeddings(self, items: List[Dict[str, Any]]) -> bool:
        """
        Store embeddings in the vector database with full functionality.

        Args:
            items: List of dictionaries containing content, metadata, and embeddings

        Returns:
            True if successful, False otherwise
        """
        if not self.index or not items:
            self.logger.warning("No index available or empty items list")
            return False

        # Filter and validate items
        valid_items = self._validate_items(items)
        if not valid_items:
            self.logger.warning("No valid embeddings to store")
            return False

        self.logger.info(f"Storing {len(valid_items)} embeddings in Pinecone")
        start_time = time.time()

        try:
            # Process in batches
            total_batches = (len(valid_items) + self.batch_size - 1) // self.batch_size
            successful_batches = 0

            for i in range(0, len(valid_items), self.batch_size):
                batch_num = (i // self.batch_size) + 1
                batch = valid_items[i : i + self.batch_size]

                self.logger.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} vectors)"
                )

                success = self._store_batch(batch)
                if success:
                    successful_batches += 1
                    self.stats["vectors_stored"] += len(batch)
                else:
                    self.stats["failed_operations"] += 1
                    self.logger.error(f" Batch {batch_num} failed")

                # Rate limiting between batches
                if i + self.batch_size < len(valid_items):
                    time.sleep(0.1)

            self.stats["batch_operations"] += total_batches
            processing_time = time.time() - start_time

            success_rate = successful_batches / total_batches * 100
            self.logger.info(
                f"Storage completed: {successful_batches}/{total_batches} batches successful ({success_rate:.1f}%) in {processing_time:.2f}s"
            )

            return successful_batches > 0

        except Exception as e:
            self.stats["failed_operations"] += 1
            raise VectorStorageError(f"Failed to store embeddings: {str(e)}")

    def _validate_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and prepare items for storage.

        Args:
            items: List of items to validate

        Returns:
            List of valid items
        """
        valid_items = []

        for i, item in enumerate(items):
            try:
                # Check required fields
                if not isinstance(item, dict):
                    self.logger.warning(f"Item {i} is not a dictionary")
                    continue

                if "embedding" not in item or not item["embedding"]:
                    self.logger.warning(f"Item {i} missing embedding")
                    continue

                embedding = item["embedding"]
                if not isinstance(embedding, list) or len(embedding) != self.dimension:
                    self.logger.warning(
                        f"Item {i} has invalid embedding dimension: {len(embedding)} != {self.dimension}"
                    )
                    continue

                # Prepare item
                processed_item = self._prepare_item_for_storage(item, i)
                valid_items.append(processed_item)

            except Exception as e:
                self.logger.warning(f"Error validating item {i}: {str(e)}")
                continue

        return valid_items

    def _prepare_item_for_storage(
        self, item: Dict[str, Any], index: int
    ) -> Dict[str, Any]:
        """
        Prepare item for Pinecone storage.

        Args:
            item: Item to prepare
            index: Item index for ID generation

        Returns:
            Prepared item
        """
        # 🆔 Generate unique ID
        item_id = item.get("id")
        if not item_id:
            # Create ID from content hash + timestamp
            content = item.get("content", "")
            timestamp = str(int(time.time() * 1000))
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            item_id = f"doc_{content_hash}_{timestamp}_{index}"

        # Prepare metadata
        metadata = item.get("metadata", {}).copy()

        # Add essential fields to metadata
        metadata.update(
            {
                "content_preview": item.get("content", "")[:500],  # First 500 chars
                "content_length": len(item.get("content", "")),
                "stored_at": datetime.now().isoformat(),
                "source": item.get("source", "unknown"),
                "document_type": item.get("document_type", "text"),
            }
        )

        # Ensure metadata size limit
        metadata = self._truncate_metadata(metadata)

        return {"id": item_id, "values": item["embedding"], "metadata": metadata}

    def _truncate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Truncate metadata to fit Pinecone size limits.

        Args:
            metadata: Original metadata

        Returns:
            Truncated metadata
        """
        import json

        # 📏 Check current size
        metadata_str = json.dumps(metadata, default=str)
        if len(metadata_str.encode()) <= self.max_metadata_size:
            return metadata

        # Truncate large fields
        truncated = metadata.copy()

        # Truncate text fields progressively
        text_fields = ["content_preview", "text", "description", "summary"]
        for field in text_fields:
            if field in truncated:
                while (
                    len(json.dumps(truncated, default=str).encode())
                    > self.max_metadata_size
                ):
                    current_length = len(str(truncated[field]))
                    if current_length <= 50:
                        break
                    truncated[field] = (
                        str(truncated[field])[: current_length // 2] + "..."
                    )

        return truncated

    def _store_batch(self, batch: List[Dict[str, Any]]) -> bool:
        """
        Store a batch of embeddings in Pinecone.

        Args:
            batch: List of prepared items

        Returns:
            True if successful
        """
        try:
            # Upsert vectors to Pinecone
            upsert_response = self.index.upsert(
                vectors=batch, timeout=self.upsert_timeout
            )

            # Verify upsert success
            if hasattr(upsert_response, "upserted_count"):
                expected_count = len(batch)
                actual_count = upsert_response.upserted_count

                if actual_count != expected_count:
                    self.logger.warning(
                        f"Upsert count mismatch: {actual_count}/{expected_count}"
                    )
                    return False

            self.logger.info(f"Successfully stored batch of {len(batch)} vectors")
            return True

        except Exception as e:
            self.logger.error(f" Error storing batch: {str(e)}")
            return False

    @error_handler(ErrorType.VECTOR_STORAGE)
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors with advanced filtering.

        Args:
            query_embedding: Query vector to search for
            top_k: Number of results to return
            filter: Optional metadata filter
            include_metadata: Whether to include metadata in results
            include_values: Whether to include vector values in results

        Returns:
            List of search results with scores and metadata
        """
        if not self.index or not query_embedding:
            self.logger.warning("No index available or empty query embedding")
            return []

        # Validate query embedding
        if len(query_embedding) != self.dimension:
            raise VectorStorageError(
                f"Query embedding dimension {len(query_embedding)} != {self.dimension}"
            )

        self.logger.info(f"Searching for similar vectors (top_k={top_k})")
        start_time = time.time()

        try:
            # Perform similarity search
            search_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter,
                include_metadata=include_metadata,
                include_values=include_values,
                timeout=self.query_timeout,
            )

            # Process results
            results = []
            if hasattr(search_response, "matches"):
                for match in search_response.matches:
                    result = {
                        "id": match.id,
                        "score": float(match.score),
                    }

                    if include_metadata and hasattr(match, "metadata"):
                        result["metadata"] = (
                            dict(match.metadata) if match.metadata else {}
                        )

                    if include_values and hasattr(match, "values"):
                        result["values"] = match.values

                    results.append(result)

            self.stats["vectors_queried"] += len(results)
            search_time = time.time() - start_time

            self.logger.info(
                f"Search completed: {len(results)} results in {search_time:.3f}s"
            )
            return results

        except Exception as e:
            self.stats["failed_operations"] += 1
            raise VectorStorageError(f"Search failed: {str(e)}")

    @error_handler(ErrorType.VECTOR_STORAGE)
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        delete_all: bool = False,
    ) -> bool:
        """
        Delete vectors from the database.

        Args:
            ids: Optional list of vector IDs to delete
            filter: Optional metadata filter for vectors to delete
            delete_all: Whether to delete all vectors

        Returns:
            True if successful
        """
        if not self.index:
            self.logger.warning("No index available")
            return False

        try:
            if delete_all:
                # Delete all vectors
                self.index.delete(delete_all=True)
                self.logger.info("Deleted all vectors from index")
                self.stats["vectors_deleted"] += 1  # Approximate

            elif ids:
                # Delete by IDs
                self.index.delete(ids=ids)
                self.logger.info(f"Deleted {len(ids)} vectors by ID")
                self.stats["vectors_deleted"] += len(ids)

            elif filter:
                # Delete by filter
                self.index.delete(filter=filter)
                self.logger.info(f"Deleted vectors by filter: {filter}")
                self.stats["vectors_deleted"] += 1  # Approximate

            else:
                self.logger.warning("No deletion criteria provided")
                return False

            return True

        except Exception as e:
            self.stats["failed_operations"] += 1
            raise VectorStorageError(f"Delete operation failed: {str(e)}")

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive index statistics.

        Returns:
            Dictionary with index statistics
        """
        if not self.index:
            return {}

        try:
            # Get Pinecone index stats
            pinecone_stats = self.index.describe_index_stats()

            # Combine with internal stats
            runtime = datetime.now() - self.stats["start_time"]

            return {
                "pinecone_stats": {
                    "total_vector_count": pinecone_stats.total_vector_count,
                    "dimension": pinecone_stats.dimension,
                    "index_fullness": pinecone_stats.index_fullness,
                    "namespaces": (
                        dict(pinecone_stats.namespaces)
                        if pinecone_stats.namespaces
                        else {}
                    ),
                },
                "internal_stats": {
                    **self.stats,
                    "runtime_seconds": runtime.total_seconds(),
                    "avg_vectors_per_batch": (
                        self.stats["vectors_stored"]
                        / max(1, self.stats["batch_operations"])
                    ),
                    "success_rate": (
                        (
                            self.stats["batch_operations"]
                            - self.stats["failed_operations"]
                        )
                        / max(1, self.stats["batch_operations"])
                        * 100
                    ),
                },
                "configuration": {
                    "index_name": self.index_name,
                    "dimension": self.dimension,
                    "metric": self.metric,
                    "batch_size": self.batch_size,
                },
            }

        except Exception as e:
            self.logger.error(f" Error getting index stats: {str(e)}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector database.

        Returns:
            Health check results
        """
        health = {
            "status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
        }

        try:
            # Check API connection
            if self.pc:
                health["checks"]["api_connection"] = "Connected"
            else:
                health["checks"]["api_connection"] = " Not connected"
                health["status"] = "unhealthy"
                return health

            # Check index availability
            if self.index:
                health["checks"]["index_available"] = "Available"
            else:
                health["checks"]["index_available"] = " Not available"
                health["status"] = "unhealthy"
                return health

            # Test query operation
            try:
                test_vector = [0.1] * self.dimension
                self.index.query(vector=test_vector, top_k=1, timeout=5)
                health["checks"]["query_operation"] = "Working"
            except Exception as e:
                health["checks"]["query_operation"] = f" Failed: {str(e)}"
                health["status"] = "degraded"

            # Check index stats
            try:
                stats = self.index.describe_index_stats()
                health["checks"]["index_stats"] = f"{stats.total_vector_count} vectors"
            except Exception as e:
                health["checks"]["index_stats"] = f" Failed: {str(e)}"

            # 🎯 Overall status
            if health["status"] == "unknown":
                health["status"] = "healthy"

        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)

        return health

    def reset_stats(self):
        """Reset internal statistics."""
        self.stats = {
            "vectors_stored": 0,
            "vectors_queried": 0,
            "vectors_deleted": 0,
            "batch_operations": 0,
            "failed_operations": 0,
            "start_time": datetime.now(),
        }
        self.logger.info("Statistics reset")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get simplified stats for UI display.

        Returns:
            Dictionary with basic statistics
        """
        try:
            if not self.index:
                return {"total_vectors": 0, "status": "disconnected"}

            # Get Pinecone stats
            pinecone_stats = self.index.describe_index_stats()

            return {
                "total_vectors": pinecone_stats.total_vector_count,
                "dimension": pinecone_stats.dimension,
                "index_fullness": pinecone_stats.index_fullness,
                "status": "connected",
            }
        except Exception as e:
            self.logger.warning(f"Could not get stats: {e}")
            return {"total_vectors": 0, "status": "error", "error": str(e)}

    def get_unique_sources(self) -> List[Dict[str, Any]]:
        """
        Get unique sources from stored vectors.

        Returns:
            List of unique sources with metadata
        """
        try:
            if not self.index:
                return []

            # This is a simplified approach - in a real implementation,
            # you might want to maintain a separate metadata index
            # For now, we'll return mock data based on what might be stored

            # Try to get some sample vectors to extract sources
            test_vector = [0.1] * self.dimension
            results = self.index.query(
                vector=test_vector,
                top_k=100,  # Get more results to find unique sources
                include_metadata=True,
            )

            sources = {}
            for match in results.matches:
                if hasattr(match, "metadata") and match.metadata:
                    source = match.metadata.get("source", "Unknown")
                    if source not in sources:
                        sources[source] = {
                            "source": source,
                            "chunk_count": 1,
                            "added_date": match.metadata.get("stored_at", "Unknown"),
                        }
                    else:
                        sources[source]["chunk_count"] += 1

            return list(sources.values())

        except Exception as e:
            self.logger.warning(f"Could not get unique sources: {e}")
            return []

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the vector database.

        Returns:
            List of document information
        """
        try:
            # Get unique sources and format as documents
            sources = self.get_unique_sources()
            documents = []

            for source_info in sources:
                documents.append(
                    {
                        "name": source_info["source"],
                        "chunks": source_info["chunk_count"],
                        "date": source_info["added_date"],
                    }
                )

            return documents

        except Exception as e:
            self.logger.warning(f"Could not list documents: {e}")
            return []
