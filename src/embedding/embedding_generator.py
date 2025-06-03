"""
Embedding Generator Module

This module is responsible for generating vector embeddings for text chunks
using Gemini Embedding v3 with complete API integration.

Technology: Gemini Embedding v3 (gemini-embedding-exp-03-07)
"""

import logging
import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import json

# Import Gemini API and caching libraries
try:
    import google.generativeai as genai
    from cachetools import TTLCache
except ImportError as e:
    logging.warning(f"Some embedding libraries are not installed: {e}")

from utils.error_handler import EmbeddingError, error_handler, ErrorType


class EmbeddingGenerator:
    """
    Generates vector embeddings for text chunks using Gemini Embedding v3 with full functionality.

    Features:
    - Gemini Embedding v3 API integration
    - Batch processing with rate limiting
    - Intelligent retry logic with exponential backoff
    - Embedding caching mechanism
    - Cost optimization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EmbeddingGenerator with configuration.

        Args:
            config: Configuration dictionary with API parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # API Configuration
        self.api_key = self.config.get("api_key", os.environ.get("GEMINI_API_KEY"))
        self.model = self.config.get("model", "gemini-embedding-exp-03-07")
        self.batch_size = self.config.get("batch_size", 5)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1)

        # Performance settings
        self.rate_limit_delay = self.config.get("rate_limit_delay", 0.1)
        self.max_text_length = self.config.get(
            "max_text_length", 8192
        )  # ✨ 8K token limit for latest model
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour

        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_tokens_processed": 0,
            "start_time": datetime.now(),
        }

        # Initialize cache
        if self.enable_caching:
            self.cache = TTLCache(maxsize=1000, ttl=self.cache_ttl)
        else:
            self.cache = None

        # Validate and initialize API client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Gemini API client with validation."""
        if not self.api_key:
            self.logger.warning(
                "No Gemini API key provided. Embeddings will not be generated."
            )
            self.client = None
            return

        try:
            # Configure Gemini API
            genai.configure(api_key=self.api_key)

            # Test API connection
            self._test_api_connection()

            self.client = genai
            self.logger.info("Gemini API client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API client: {str(e)}")
            self.client = None

    def _test_api_connection(self):
        """Test API connection with a simple request."""
        try:
            # Test with a simple embedding request
            test_result = genai.embed_content(
                model=self.model,
                content="test connection",
                task_type="retrieval_document",
            )

            if not test_result.get("embedding"):
                raise Exception("No embedding returned from test request")

            self.logger.info("API connection test successful")

        except Exception as e:
            raise EmbeddingError(f"API connection test failed: {str(e)}")

    @error_handler(ErrorType.EMBEDDING_GENERATION)
    def generate_embeddings(self, texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of text chunks with full functionality.

        Args:
            texts: List of dictionaries containing text chunks and metadata
                Each dict should have 'content' and 'metadata' keys

        Returns:
            List of dictionaries with original content, metadata, and embeddings
        """
        if not self.client or not texts:
            self.logger.warning("No API client or empty text list")
            return texts

        self.logger.info(f"Generating embeddings for {len(texts)} text chunks")
        start_time = time.time()

        # Filter and validate texts
        valid_texts = self._validate_texts(texts)
        if not valid_texts:
            self.logger.warning("No valid texts to process")
            return texts

        # Process in batches to respect API limits
        results = []
        total_batches = (len(valid_texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(valid_texts), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch = valid_texts[i : i + self.batch_size]

            self.logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)"
            )

            try:
                batch_results = self._process_batch(batch)
                results.extend(batch_results)

                # Rate limiting between batches
                if i + self.batch_size < len(valid_texts):
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                self.logger.error(f"Batch {batch_num} failed: {str(e)}")
                # Add original items without embeddings
                for item in batch:
                    item_copy = item.copy()
                    item_copy["embedding"] = []
                    item_copy["embedding_error"] = str(e)
                    results.append(item_copy)

        # Update statistics
        processing_time = time.time() - start_time
        self.logger.info(f"Embedding generation completed in {processing_time:.2f}s")

        return results

    def _validate_texts(self, texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and filter text inputs.

        Args:
            texts: List of text dictionaries

        Returns:
            List of valid text dictionaries
        """
        valid_texts = []

        for i, item in enumerate(texts):
            if not isinstance(item, dict) or "content" not in item:
                self.logger.warning(f"Invalid item at index {i}: missing 'content' key")
                continue

            content = item["content"]
            if not content or not isinstance(content, str):
                self.logger.warning(
                    f"Invalid content at index {i}: empty or non-string"
                )
                continue

            # Truncate if too long
            if len(content) > self.max_text_length:
                self.logger.warning(
                    f"Truncating text at index {i}: {len(content)} -> {self.max_text_length} chars"
                )
                item = item.copy()
                item["content"] = content[: self.max_text_length]
                item["metadata"] = item.get("metadata", {})
                item["metadata"]["truncated"] = True
                item["metadata"]["original_length"] = len(content)

            valid_texts.append(item)

        return valid_texts

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of text chunks to generate embeddings.

        Args:
            batch: List of dictionaries containing text chunks and metadata

        Returns:
            List of dictionaries with original content, metadata, and embeddings
        """
        # Extract content and check cache
        contents = []
        cache_results = {}

        for i, item in enumerate(batch):
            content = item["content"]

            # Check cache first
            if self.cache is not None:
                cache_key = self._get_cache_key(content)
                if cache_key in self.cache:
                    cache_results[i] = self.cache[cache_key]
                    self.stats["cache_hits"] += 1
                    continue

            contents.append((i, content))

        # Generate embeddings for non-cached content
        embeddings_map = {}
        if contents:
            content_texts = [content for _, content in contents]
            embeddings = self._generate_with_retry(content_texts)

            # Map embeddings back to indices
            for j, (original_index, content) in enumerate(contents):
                if j < len(embeddings):
                    embedding = embeddings[j]
                    embeddings_map[original_index] = embedding

                    # Cache the result
                    if self.cache is not None:
                        cache_key = self._get_cache_key(content)
                        self.cache[cache_key] = embedding

        # 🔗 Combine results
        results = []
        for i, item in enumerate(batch):
            result = item.copy()

            # Add embedding from cache or new generation
            if i in cache_results:
                result["embedding"] = cache_results[i]
                result["embedding_source"] = "cache"
            elif i in embeddings_map:
                result["embedding"] = embeddings_map[i]
                result["embedding_source"] = "api"
            else:
                result["embedding"] = []
                result["embedding_source"] = "failed"
                self.logger.warning(f"Missing embedding for batch item {i}")

            # Add embedding metadata
            if result["embedding"]:
                result["metadata"] = result.get("metadata", {})
                result["metadata"].update(
                    {
                        "embedding_model": self.model,
                        "embedding_dimension": len(result["embedding"]),
                        "embedding_generated_at": datetime.now().isoformat(),
                    }
                )

            results.append(result)

        return results

    def _generate_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings with intelligent retry logic.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        for attempt in range(self.max_retries):
            try:
                self.stats["total_requests"] += 1

                # Generate embeddings using Gemini API
                embeddings = []

                for text in texts:
                    try:
                        # Track tokens
                        self.stats["total_tokens_processed"] += len(text.split())

                        # Call Gemini API
                        result = self.client.embed_content(
                            model=self.model,
                            content=text,
                            task_type="retrieval_document",
                            title="Document chunk for RAG system",
                        )

                        if result and "embedding" in result:
                            embeddings.append(result["embedding"])
                        else:
                            self.logger.warning(
                                f"No embedding in API response for text: {text[:50]}..."
                            )
                            embeddings.append([])

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to embed individual text: {str(e)}"
                        )
                        embeddings.append([])

                self.stats["successful_requests"] += 1
                return embeddings

            except Exception as e:
                self.stats["failed_requests"] += 1
                self.logger.warning(
                    f"Embedding generation failed (attempt {attempt+1}/{self.max_retries}): {str(e)}"
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = self.retry_delay * (2**attempt) + (time.time() % 1)
                    self.logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)

        # All retries failed
        self.logger.error("All embedding generation attempts failed")
        return [[] for _ in texts]

    @error_handler(ErrorType.EMBEDDING_GENERATION)
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query string.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as a list of floats
        """
        if not self.client or not query:
            return []

        self.logger.info(f"Generating embedding for query: {query[:50]}...")

        # Check cache first
        if self.cache is not None:
            cache_key = self._get_cache_key(query, "query")
            if cache_key in self.cache:
                self.stats["cache_hits"] += 1
                return self.cache[cache_key]

        # Generate embedding
        embeddings = self._generate_with_retry([query])
        embedding = embeddings[0] if embeddings else []

        # Cache the result
        if embedding and self.cache is not None:
            cache_key = self._get_cache_key(query, "query")
            self.cache[cache_key] = embedding

        return embedding

    def _get_cache_key(self, text: str, prefix: str = "doc") -> str:
        """
        Generate cache key for text.

        Args:
            text: Text content
            prefix: Key prefix

        Returns:
            Cache key string
        """
        # 🔐 Create hash of text + model for unique key
        content_hash = hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
        return f"{prefix}:{content_hash}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get embedding generation statistics.

        Returns:
            Dictionary with statistics
        """
        runtime = datetime.now() - self.stats["start_time"]

        return {
            **self.stats,
            "runtime_seconds": runtime.total_seconds(),
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(1, self.stats["total_requests"]) * 100
            ),
            "success_rate": (
                self.stats["successful_requests"]
                / max(1, self.stats["total_requests"])
                * 100
            ),
            "avg_tokens_per_request": (
                self.stats["total_tokens_processed"]
                / max(1, self.stats["total_requests"])
            ),
            "cache_size": len(self.cache) if self.cache else 0,
            "model": self.model,
            "batch_size": self.batch_size,
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()
            self.logger.info("Embedding cache cleared")

    def warm_up_cache(self, sample_texts: List[str]):
        """
        🔥 Warm up the cache with sample texts.

        Args:
            sample_texts: List of sample texts to pre-generate embeddings
        """
        if not sample_texts:
            return

        self.logger.info(f"🔥 Warming up cache with {len(sample_texts)} sample texts")

        sample_items = [{"content": text, "metadata": {}} for text in sample_texts]
        self.generate_embeddings(sample_items)

        self.logger.info("Cache warm-up completed")
