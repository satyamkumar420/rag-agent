"""
Configuration Manager Module

This module handles loading and managing configuration settings
for the RAG AI system.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """
    Manages configuration settings for the RAG AI system.

    Features:
    - YAML configuration file loading
    - Environment variable override
    - Default configuration values
    - Configuration validation
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigManager.

        Args:
            config_path: Path to the configuration file (defaults to config/config.yaml)
        """
        self.logger = logging.getLogger(__name__)

        # Set default config path
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "config", "config.yaml"
            )

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file and environment variables.

        Returns:
            Configuration dictionary
        """
        # Start with default configuration
        config = self._get_default_config()

        # Load from YAML file if it exists
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    file_config = yaml.safe_load(f) or {}
                    config = self._merge_configs(config, file_config)
                self.logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load config file {self.config_path}: {str(e)}"
                )
        else:
            self.logger.warning(f"Config file not found: {self.config_path}")

        # Override with environment variables
        config = self._apply_env_overrides(config)

        # Validate configuration
        self._validate_config(config)

        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values.

        Returns:
            Default configuration dictionary
        """
        return {
            "api_keys": {
                "gemini_api_key": "",
                "pinecone_api_key": "",
                "openai_api_key": "",
            },
            "vector_db": {
                "provider": "pinecone",
                "index_name": "rag-ai-index",
                "dimension": 3072,  # ✅ Fixed: Match Gemini embedding dimension
                "metric": "cosine",
                "environment": "us-west1-gcp",
            },
            "embedding": {
                "model": "gemini-embedding-exp-03-07",
                "batch_size": 5,
                "max_retries": 3,
                "retry_delay": 1,
            },
            "document_processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "min_chunk_size": 100,
                "max_file_size_mb": 50,
            },
            "url_processing": {
                "max_depth": 1,
                "follow_links": True,
                "max_pages": 10,
                "timeout": 10,
            },
            "rag": {
                "top_k": 5,
                "similarity_threshold": 0.7,
                "max_context_length": 4000,
                "model": "gpt-3.5-turbo",
                "max_tokens": 500,
                "temperature": 0.7,
            },
            "ui": {
                "title": "AI Embedded Knowledge Agent",
                "description": "Upload documents or provide URLs to build your knowledge base, then ask questions!",
                "theme": "default",
                "share": False,
                "port": 7860,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

    def _merge_configs(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with environment overrides applied
        """
        # API Keys
        if os.environ.get("GEMINI_API_KEY"):
            config["api_keys"]["gemini_api_key"] = os.environ["GEMINI_API_KEY"]

        if os.environ.get("PINECONE_API_KEY"):
            config["api_keys"]["pinecone_api_key"] = os.environ["PINECONE_API_KEY"]

        if os.environ.get("OPENAI_API_KEY"):
            config["api_keys"]["openai_api_key"] = os.environ["OPENAI_API_KEY"]

        # Pinecone settings
        if os.environ.get("PINECONE_ENVIRONMENT"):
            config["vector_db"]["environment"] = os.environ["PINECONE_ENVIRONMENT"]

        if os.environ.get("PINECONE_INDEX_NAME"):
            config["vector_db"]["index_name"] = os.environ["PINECONE_INDEX_NAME"]

        # UI settings
        if os.environ.get("GRADIO_SHARE"):
            config["ui"]["share"] = os.environ["GRADIO_SHARE"].lower() == "true"

        if os.environ.get("PORT"):
            try:
                config["ui"]["port"] = int(os.environ["PORT"])
            except ValueError:
                self.logger.warning(f"Invalid PORT value: {os.environ['PORT']}")

        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration values.

        Args:
            config: Configuration dictionary to validate
        """
        # Check required API keys
        if not config["api_keys"]["gemini_api_key"]:
            self.logger.warning("Gemini API key not configured")

        if not config["api_keys"]["pinecone_api_key"]:
            self.logger.warning("Pinecone API key not configured")

        # Validate numeric values
        if config["document_processing"]["chunk_size"] <= 0:
            raise ValueError("chunk_size must be positive")

        if config["rag"]["top_k"] <= 0:
            raise ValueError("top_k must be positive")

        if not 0 <= config["rag"]["similarity_threshold"] <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'vector_db.index_name')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'vector_db.index_name')
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Section name

        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, {})

    def reload(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        self.logger.info("Configuration reloaded")
