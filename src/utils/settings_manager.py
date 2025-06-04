"""
Settings Manager Module

This module provides secure environment variable management with UI integration,
supporting both cache and .env file storage options.

Features:
- 🔐 Secure API key handling with masking
- ⚡ Real-time validation and testing
- 💾 Dual storage backends (cache + .env file)
- 🛡️ Input sanitization and validation
- 🔄 Live system updates
"""

import os
import re
import logging
import json
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime
import tempfile


class SettingsManager:
    """
    Manages environment variables with secure storage and validation.

    Features:
    - Secure API key masking and validation
    - Real-time connection testing
    - Cache and .env file storage options
    - Integration with existing ConfigManager
    """

    def __init__(self, config_manager=None):
        """
        Initialize the SettingsManager.

        Args:
            config_manager: Optional ConfigManager instance for integration
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager

        # 🔧 Cache storage for temporary settings
        self._cache_storage = {}

        # 📁 Project root for .env file
        self.project_root = Path(__file__).parent.parent.parent
        self.env_file_path = self.project_root / ".env"

        # 🛡️ Supported environment variables with validation rules
        self.supported_env_vars = {
            "GEMINI_API_KEY": {
                "required": True,
                "description": "Google Gemini API Key for embeddings and LLM",
                "format": r"^AIzaSy[A-Za-z0-9_-]{33}$",
                "mask": True,
                "test_function": self._test_gemini_connection,
                "placeholder": "AIzaSy...",
                "help_url": "https://aistudio.google.com/",
            },
            "PINECONE_API_KEY": {
                "required": False,
                "description": "Pinecone API Key for vector database",
                "format": r"^pc-[A-Za-z0-9]{32}$",
                "mask": True,
                "test_function": self._test_pinecone_connection,
                "placeholder": "pc-...",
                "help_url": "https://www.pinecone.io/",
            },
            "OPENAI_API_KEY": {
                "required": False,
                "description": "OpenAI API Key for alternative LLM",
                "format": r"^sk-[A-Za-z0-9]{48}$",
                "mask": True,
                "test_function": self._test_openai_connection,
                "placeholder": "sk-...",
                "help_url": "https://platform.openai.com/api-keys",
            },
            "TAVILY_API_KEY": {
                "required": False,
                "description": "Tavily API Key for live web search",
                "format": r"^tvly-[A-Za-z0-9-]{20,50}$",
                "mask": True,
                "test_function": self._test_tavily_connection,
                "placeholder": "tvly-dev-...",
                "help_url": "https://app.tavily.com/sign-in",
            },
            "PINECONE_ENVIRONMENT": {
                "required": False,
                "description": "Pinecone environment region",
                "format": r"^[a-z0-9-]+$",
                "mask": False,
                "default": "us-east-1",
                "placeholder": "us-east-1",
                "options": [
                    "us-east-1",
                    "us-west1-gcp",
                    "eu-west1-gcp",
                    "asia-southeast1-gcp",
                ],
            },
            "PINECONE_INDEX_NAME": {
                "required": False,
                "description": "Pinecone index name",
                "format": r"^[a-z0-9-]+$",
                "mask": False,
                "default": "rag-ai-index",
                "placeholder": "rag-ai-index",
            },
            "GRADIO_SHARE": {
                "required": False,
                "description": "Enable Gradio public sharing",
                "format": r"^(true|false)$",
                "mask": False,
                "default": "false",
                "options": ["true", "false"],
            },
            "PORT": {
                "required": False,
                "description": "Server port number",
                "format": r"^[1-9][0-9]{3,4}$",
                "mask": False,
                "default": "7860",
                "placeholder": "7860",
            },
        }

        self.logger.info("SettingsManager initialized successfully")

    def get_current_settings(self) -> Dict[str, Any]:
        """
        Get current environment variable settings with status.

        Returns:
            Dictionary with current settings and their status
        """
        settings = {}

        for var_name, config in self.supported_env_vars.items():
            # 🔍 Get value from cache, environment, or default
            value = self._get_env_value(var_name)

            settings[var_name] = {
                "value": (
                    self._mask_value(value, config.get("mask", False)) if value else ""
                ),
                "raw_value": value or "",
                "is_set": bool(value),
                "is_valid": (
                    self._validate_format(value, config.get("format"))
                    if value
                    else False
                ),
                "is_required": config.get("required", False),
                "description": config.get("description", ""),
                "placeholder": config.get("placeholder", ""),
                "help_url": config.get("help_url", ""),
                "options": config.get("options", []),
                "source": self._get_value_source(var_name),
                "last_tested": self._cache_storage.get(f"{var_name}_last_tested"),
                "test_status": self._cache_storage.get(
                    f"{var_name}_test_status", "untested"
                ),
            }

        return settings

    def update_setting(
        self, var_name: str, value: str, storage_type: str = "cache"
    ) -> Dict[str, Any]:
        """
        Update an environment variable setting.

        Args:
            var_name: Environment variable name
            value: New value
            storage_type: "cache" or "env_file"

        Returns:
            Dictionary with operation result
        """
        try:
            if var_name not in self.supported_env_vars:
                return {
                    "success": False,
                    "error": f"Unsupported environment variable: {var_name}",
                    "status": "❌ Invalid variable",
                }

            config = self.supported_env_vars[var_name]

            # 🛡️ Validate format
            if value and not self._validate_format(value, config.get("format")):
                return {
                    "success": False,
                    "error": f"Invalid format for {var_name}",
                    "status": "❌ Invalid format",
                    "expected_format": config.get("placeholder", ""),
                }

            # 💾 Store based on storage type
            if storage_type == "cache":
                self._cache_storage[var_name] = value
                os.environ[var_name] = value  # ⚡ Update current session
                status_msg = "💾 Saved to cache"
            elif storage_type == "env_file":
                self._save_to_env_file(var_name, value)
                os.environ[var_name] = value  # ⚡ Update current session
                status_msg = "📁 Saved to .env file"
            else:
                return {
                    "success": False,
                    "error": f"Invalid storage type: {storage_type}",
                    "status": "❌ Invalid storage type",
                }

            # 🔄 Update config manager if available
            if self.config_manager:
                try:
                    self.config_manager.reload()
                except Exception as e:
                    self.logger.warning(f"Could not reload config manager: {e}")

            self.logger.info(f"Updated {var_name} via {storage_type}")

            return {
                "success": True,
                "status": f" {status_msg}",
                "value": self._mask_value(value, config.get("mask", False)),
                "storage_type": storage_type,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error updating {var_name}: {e}")
            return {"success": False, "error": str(e), "status": " Update failed"}

    def test_connection(self, var_name: str) -> Dict[str, Any]:
        """
        Test API connection for a given environment variable.

        Args:
            var_name: Environment variable name

        Returns:
            Dictionary with test results
        """
        try:
            if var_name not in self.supported_env_vars:
                return {
                    "success": False,
                    "error": f"Cannot test {var_name}: not supported",
                    "status": "❌ Not testable",
                }

            config = self.supported_env_vars[var_name]
            test_function = config.get("test_function")

            if not test_function:
                return {
                    "success": False,
                    "error": f"No test function available for {var_name}",
                    "status": "⚠️ No test available",
                }

            value = self._get_env_value(var_name)
            if not value:
                return {
                    "success": False,
                    "error": f"{var_name} is not set",
                    "status": "❌ Not configured",
                }

            # 🧪 Run the test
            self.logger.info(f"Testing connection for {var_name}")
            test_result = test_function(value)

            # 📊 Cache test results
            timestamp = datetime.now().isoformat()
            self._cache_storage[f"{var_name}_last_tested"] = timestamp
            self._cache_storage[f"{var_name}_test_status"] = (
                "success" if test_result["success"] else "failed"
            )

            return {**test_result, "timestamp": timestamp, "variable": var_name}

        except Exception as e:
            self.logger.error(f"Error testing {var_name}: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "status": "❌ Test failed",
                "timestamp": datetime.now().isoformat(),
            }

            # 📊 Cache failed test
            self._cache_storage[f"{var_name}_last_tested"] = error_result["timestamp"]
            self._cache_storage[f"{var_name}_test_status"] = "failed"

            return error_result

    def load_from_env_file(self) -> Dict[str, Any]:
        """
        Load settings from .env file.

        Returns:
            Dictionary with load results
        """
        try:
            if not self.env_file_path.exists():
                return {
                    "success": False,
                    "error": ".env file not found",
                    "status": "📁 No .env file found",
                    "loaded_count": 0,
                }

            loaded_vars = []

            with open(self.env_file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        try:
                            key, value = line.split("=", 1)
                            key = key.strip()
                            value = value.strip().strip("\"'")  # Remove quotes

                            if key in self.supported_env_vars:
                                os.environ[key] = value
                                loaded_vars.append(key)
                        except Exception as e:
                            self.logger.warning(
                                f"Error parsing line {line_num} in .env: {e}"
                            )

            # 🔄 Reload config manager
            if self.config_manager:
                try:
                    self.config_manager.reload()
                except Exception as e:
                    self.logger.warning(f"Could not reload config manager: {e}")

            return {
                "success": True,
                "status": f" Loaded {len(loaded_vars)} variables from .env",
                "loaded_count": len(loaded_vars),
                "loaded_variables": loaded_vars,
            }

        except Exception as e:
            self.logger.error(f"Error loading from .env file: {e}")
            return {
                "success": False,
                "error": str(e),
                "status": " Failed to load .env file",
                "loaded_count": 0,
            }

    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear cached settings.

        Returns:
            Dictionary with operation result
        """
        try:
            # 🗑️ Clear cache but preserve test results
            cached_vars = [
                key
                for key in self._cache_storage.keys()
                if key in self.supported_env_vars
            ]

            for var in cached_vars:
                if var in self._cache_storage:
                    del self._cache_storage[var]
                    # Remove from current environment if it was cached
                    if var in os.environ:
                        del os.environ[var]

            return {
                "success": True,
                "status": f"🗑️ Cleared {len(cached_vars)} cached variables",
                "cleared_count": len(cached_vars),
            }

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return {
                "success": False,
                "error": str(e),
                "status": " Failed to clear cache",
            }

    def export_settings(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Export current settings for backup/sharing.

        Args:
            include_sensitive: Whether to include API keys (masked)

        Returns:
            Dictionary with exported settings
        """
        try:
            settings = self.get_current_settings()
            exported = {}

            for var_name, config in settings.items():
                var_config = self.supported_env_vars[var_name]

                # 🔐 Skip sensitive data if not requested
                if var_config.get("mask", False) and not include_sensitive:
                    continue

                exported[var_name] = {
                    "value": (
                        config["value"] if include_sensitive else config["raw_value"]
                    ),
                    "is_set": config["is_set"],
                    "source": config["source"],
                    "description": config["description"],
                }

            return {
                "success": True,
                "settings": exported,
                "export_timestamp": datetime.now().isoformat(),
                "include_sensitive": include_sensitive,
            }

        except Exception as e:
            self.logger.error(f"Error exporting settings: {e}")
            return {"success": False, "error": str(e)}

    # 🔧 Private helper methods

    def _get_env_value(self, var_name: str) -> Optional[str]:
        """Get environment variable value from cache or environment."""
        # Priority: cache > environment > default
        if var_name in self._cache_storage:
            return self._cache_storage[var_name]

        env_value = os.environ.get(var_name)
        if env_value:
            return env_value

        return self.supported_env_vars[var_name].get("default")

    def _get_value_source(self, var_name: str) -> str:
        """Determine the source of an environment variable value."""
        if var_name in self._cache_storage:
            return "cache"
        elif os.environ.get(var_name):
            return "environment"
        elif self.supported_env_vars[var_name].get("default"):
            return "default"
        else:
            return "unset"

    def _mask_value(self, value: str, should_mask: bool) -> str:
        """Mask sensitive values for display."""
        if not value or not should_mask:
            return value

        if len(value) <= 8:
            return "*" * len(value)

        return value[:4] + "*" * (len(value) - 8) + value[-4:]

    def _validate_format(self, value: str, format_pattern: Optional[str]) -> bool:
        """Validate value against format pattern."""
        if not format_pattern or not value:
            return True

        try:
            return bool(re.match(format_pattern, value))
        except Exception:
            return False

    def _save_to_env_file(self, var_name: str, value: str):
        """Save environment variable to .env file."""
        env_vars = {}

        # 📖 Read existing .env file
        if self.env_file_path.exists():
            with open(self.env_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        try:
                            key, val = line.split("=", 1)
                            env_vars[key.strip()] = val.strip().strip("\"'")
                        except Exception as e:
                            self.logger.warning(f"Error parsing line in .env: {e}")

        # ✏️ Update the variable
        env_vars[var_name] = value

        # 💾 Write back to file
        with open(self.env_file_path, "w", encoding="utf-8") as f:
            f.write("# Environment Variables for RAG AI System\n")
            f.write(f"# Generated on {datetime.now().isoformat()}\n\n")

            for key, val in env_vars.items():
                # 🔐 Quote values that contain spaces or special characters
                if " " in val or any(char in val for char in ["$", '"', "'"]):
                    f.write(f'{key}="{val}"\n')
                else:
                    f.write(f"{key}={val}\n")

    # 🧪 API Testing Functions

    def _test_gemini_connection(self, api_key: str) -> Dict[str, Any]:
        """Test Gemini API connection."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)

            # 🧪 Simple test call
            model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
            response = model.generate_content("Hello")

            if response and response.text:
                return {
                    "success": True,
                    "status": "✅ Gemini API connected",
                    "details": "API key is valid and working",
                }
            else:
                return {
                    "success": False,
                    "status": "❌ Gemini API failed",
                    "error": "No response from API",
                }

        except Exception as e:
            return {
                "success": False,
                "status": "❌ Gemini connection failed",
                "error": str(e),
            }

    def _test_pinecone_connection(self, api_key: str) -> Dict[str, Any]:
        """Test Pinecone API connection."""
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=api_key)

            # 🧪 Test by listing indexes
            indexes = pc.list_indexes()

            return {
                "success": True,
                "status": "✅ Pinecone API connected",
                "details": f"Found {len(indexes)} indexes",
            }

        except Exception as e:
            return {
                "success": False,
                "status": "❌ Pinecone connection failed",
                "error": str(e),
            }

    def _test_openai_connection(self, api_key: str) -> Dict[str, Any]:
        """Test OpenAI API connection."""
        try:
            import openai

            client = openai.OpenAI(api_key=api_key)

            # 🧪 Test with a simple completion
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )

            if response and response.choices:
                return {
                    "success": True,
                    "status": "✅ OpenAI API connected",
                    "details": "API key is valid and working",
                }
            else:
                return {
                    "success": False,
                    "status": "❌ OpenAI API failed",
                    "error": "No response from API",
                }

        except Exception as e:
            return {
                "success": False,
                "status": " OpenAI connection failed",
                "error": str(e),
            }

    def _test_tavily_connection(self, api_key: str) -> Dict[str, Any]:
        """Test Tavily API connection."""
        try:
            from tavily import TavilyClient

            # 🧪 Initialize client and test with a simple search
            client = TavilyClient(api_key=api_key)

            # Test with a minimal search query
            response = client.search(query="test", max_results=1, search_depth="basic")

            if response and isinstance(response, dict):
                return {
                    "success": True,
                    "status": "✅ Tavily API connected",
                    "details": "API key is valid and working",
                }
            else:
                return {
                    "success": False,
                    "status": "❌ Tavily API failed",
                    "error": "No valid response from API",
                }

        except Exception as e:
            return {
                "success": False,
                "status": "❌ Tavily connection failed",
                "error": str(e),
            }
