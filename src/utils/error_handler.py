"""
Error Handler Module

This module provides centralized error handling and logging
for the RAG AI system.
"""

import logging
import traceback
import functools
from typing import Any, Callable, Dict, Optional, Type, Union
from enum import Enum


class ErrorType(Enum):
    """Enumeration of error types in the system."""

    DOCUMENT_PROCESSING = "document_processing"
    URL_PROCESSING = "url_processing"
    EMBEDDING_GENERATION = "embedding_generation"
    VECTOR_STORAGE = "vector_storage"
    QUERY_PROCESSING = "query_processing"
    RESPONSE_GENERATION = "response_generation"
    API_ERROR = "api_error"
    CONFIGURATION = "configuration"
    UI_ERROR = "ui_error"
    UNKNOWN = "unknown"


class RAGError(Exception):
    """Base exception class for RAG AI system errors."""

    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RAGError.

        Args:
            message: Error message
            error_type: Type of error
            details: Additional error details
        """
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}
        self.message = message


class DocumentProcessingError(RAGError):
    """Exception for document processing errors."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if file_path:
            details["file_path"] = file_path
        super().__init__(message, ErrorType.DOCUMENT_PROCESSING, details)


class URLProcessingError(RAGError):
    """Exception for URL processing errors."""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if url:
            details["url"] = url
        super().__init__(message, ErrorType.URL_PROCESSING, details)


class EmbeddingError(RAGError):
    """Exception for embedding generation errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.EMBEDDING_GENERATION, details)


class VectorStorageError(RAGError):
    """Exception for vector storage errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.VECTOR_STORAGE, details)


class QueryProcessingError(RAGError):
    """Exception for query processing errors."""

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if query:
            details["query"] = query
        super().__init__(message, ErrorType.QUERY_PROCESSING, details)


class ResponseGenerationError(RAGError):
    """Exception for response generation errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.RESPONSE_GENERATION, details)


class APIError(RAGError):
    """Exception for API-related errors."""

    def __init__(
        self,
        message: str,
        api_name: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if api_name:
            details["api_name"] = api_name
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, ErrorType.API_ERROR, details)


class ConfigurationError(RAGError):
    """Exception for configuration errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, ErrorType.CONFIGURATION, details)


class ErrorHandler:
    """
    Centralized error handler for the RAG AI system.

    Features:
    - Error logging with context
    - Error categorization
    - Error recovery suggestions
    - Performance monitoring
    """

    def __init__(self, logger_name: str = __name__):
        """
        Initialize the ErrorHandler.

        Args:
            logger_name: Name for the logger instance
        """
        self.logger = logging.getLogger(logger_name)
        self.error_counts = {}

    def handle_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle an error with logging and context.

        Args:
            error: The exception that occurred
            context: Additional context information

        Returns:
            Dictionary containing error information
        """
        context = context or {}

        # Determine error type
        if isinstance(error, RAGError):
            error_type = error.error_type
            error_details = error.details
        else:
            error_type = ErrorType.UNKNOWN
            error_details = {}

        # Create error info
        error_info = {
            "type": error_type.value,
            "message": str(error),
            "details": error_details,
            "context": context,
            "traceback": traceback.format_exc(),
        }

        # Log the error
        self._log_error(error_info)

        # Update error counts
        self._update_error_counts(error_type)

        # Add recovery suggestions
        error_info["recovery_suggestions"] = self._get_recovery_suggestions(error_type)

        return error_info

    def _log_error(self, error_info: Dict[str, Any]) -> None:
        """
        Log error information.

        Args:
            error_info: Error information dictionary
        """
        error_type = error_info["type"]
        message = error_info["message"]
        context = error_info.get("context", {})

        log_message = f"[{error_type.upper()}] {message}"

        if context:
            context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
            log_message += f" | Context: {context_str}"

        self.logger.error(log_message)

        # Log traceback at debug level
        if error_info.get("traceback"):
            self.logger.debug(f"Traceback: {error_info['traceback']}")

    def _update_error_counts(self, error_type: ErrorType) -> None:
        """
        Update error count statistics.

        Args:
            error_type: Type of error that occurred
        """
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

    def _get_recovery_suggestions(self, error_type: ErrorType) -> list:
        """
        Get recovery suggestions for an error type.

        Args:
            error_type: Type of error

        Returns:
            List of recovery suggestions
        """
        suggestions = {
            ErrorType.DOCUMENT_PROCESSING: [
                "Check if the document format is supported",
                "Verify the document is not corrupted",
                "Ensure sufficient disk space for processing",
            ],
            ErrorType.URL_PROCESSING: [
                "Verify the URL is accessible",
                "Check internet connectivity",
                "Ensure the website allows scraping",
            ],
            ErrorType.EMBEDDING_GENERATION: [
                "Check Gemini API key configuration",
                "Verify API quota and rate limits",
                "Ensure text content is not empty",
            ],
            ErrorType.VECTOR_STORAGE: [
                "Check Pinecone API key configuration",
                "Verify Pinecone index exists",
                "Check vector dimensions match index configuration",
            ],
            ErrorType.QUERY_PROCESSING: [
                "Ensure query is not empty",
                "Check if knowledge base has content",
                "Verify embedding generation is working",
            ],
            ErrorType.RESPONSE_GENERATION: [
                "Check language model configuration",
                "Verify retrieved context is valid",
                "Ensure API keys are configured",
            ],
            ErrorType.API_ERROR: [
                "Check API key validity",
                "Verify network connectivity",
                "Check API rate limits and quotas",
            ],
            ErrorType.CONFIGURATION: [
                "Check configuration file syntax",
                "Verify all required settings are present",
                "Ensure environment variables are set",
            ],
            ErrorType.UI_ERROR: [
                "Refresh the page",
                "Check browser compatibility",
                "Verify Gradio is properly installed",
            ],
        }

        return suggestions.get(error_type, ["Contact support for assistance"])

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics.

        Returns:
            Dictionary containing error statistics
        """
        total_errors = sum(self.error_counts.values())

        return {
            "total_errors": total_errors,
            "error_counts": {
                error_type.value: count
                for error_type, count in self.error_counts.items()
            },
            "most_common_error": (
                max(self.error_counts.items(), key=lambda x: x[1])[0].value
                if self.error_counts
                else None
            ),
        }


def error_handler(
    error_type: ErrorType = ErrorType.UNKNOWN, context: Optional[Dict[str, Any]] = None
):
    """
    Decorator for automatic error handling.

    Args:
        error_type: Type of error to handle
        context: Additional context information

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or {}
                error_context.update(
                    {
                        "function": func.__name__,
                        "args": str(args)[:100],  # Truncate for logging
                        "kwargs": str(kwargs)[:100],
                    }
                )

                error_info = handler.handle_error(e, error_context)

                # Re-raise as appropriate RAG error type
                if error_type == ErrorType.DOCUMENT_PROCESSING:
                    raise DocumentProcessingError(str(e), details=error_info)
                elif error_type == ErrorType.URL_PROCESSING:
                    raise URLProcessingError(str(e), details=error_info)
                elif error_type == ErrorType.EMBEDDING_GENERATION:
                    raise EmbeddingError(str(e), details=error_info)
                elif error_type == ErrorType.VECTOR_STORAGE:
                    raise VectorStorageError(str(e), details=error_info)
                elif error_type == ErrorType.QUERY_PROCESSING:
                    raise QueryProcessingError(str(e), details=error_info)
                elif error_type == ErrorType.RESPONSE_GENERATION:
                    raise ResponseGenerationError(str(e), details=error_info)
                else:
                    raise RAGError(str(e), error_type, error_info)

        return wrapper

    return decorator
