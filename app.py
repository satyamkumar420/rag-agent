"""
AI Embedded Knowledge Agent - Main Application Entry Point

This is the main entry point for the RAG AI system that integrates all components
and launches the Gradio interface for deployment on Hugging Face.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "python-dotenv not installed. Please install it with: pip install python-dotenv"
    )

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import all components
from utils.config_manager import ConfigManager
from utils.error_handler import ErrorHandler, ErrorType
from ingestion.document_processor import DocumentProcessor
from ingestion.url_processor import URLProcessor
from ingestion.text_extractor import TextExtractor
from embedding.embedding_generator import EmbeddingGenerator
from storage.vector_db import VectorDB
from rag.optimized_query_processor import OptimizedQueryProcessor
from rag.response_generator import ResponseGenerator
from ui.gradio_app import GradioApp


class RAGSystem:
    """
    Main RAG AI system that orchestrates all components.

    This class integrates document processing, embedding generation,
    vector storage, and query processing into a unified system.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the RAG system with all components.

        Args:
            config_path: Path to configuration file
        """
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing RAG AI System...")

        # Initialize error handler
        self.error_handler = ErrorHandler()

        # Validate environment and configuration
        self._validate_environment()

        # Initialize components
        self._initialize_components()

        # Run health checks
        self._run_startup_health_checks()

        self.logger.info("RAG AI System initialized successfully! ")

    def _setup_logging(self):
        """Setup comprehensive logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO").upper())
        log_format = log_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Configure root logger with UTF-8 encoding
        import io

        utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[logging.StreamHandler(utf8_stdout)],
        )

        # Create logs directory if specified
        log_file = log_config.get("file")
        if log_file:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            # Add file handler with rotation
            try:
                from logging.handlers import RotatingFileHandler

                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=log_config.get("max_file_size_mb", 10) * 1024 * 1024,
                    backupCount=log_config.get("backup_count", 5),
                )
                file_handler.setFormatter(logging.Formatter(log_format))
                logging.getLogger().addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Could not setup file logging: {e}")

    def _validate_environment(self):
        """Validate environment variables and configuration."""
        self.logger.info("Validating environment...")

        # Check required API keys
        required_keys = ["GEMINI_API_KEY"]
        optional_keys = ["PINECONE_API_KEY", "OPENAI_API_KEY"]

        missing_required = []
        for key in required_keys:
            if not os.getenv(key):
                missing_required.append(key)

        if missing_required:
            self.logger.error(
                f" Missing required environment variables: {missing_required}"
            )
            self.logger.error(
                "Please set the required API keys as environment variables"
            )
            # Don't raise error in demo mode, just warn
            self.logger.warning("Running in demo mode with limited functionality")

        # Check optional keys
        missing_optional = []
        for key in optional_keys:
            if not os.getenv(key):
                missing_optional.append(key)

        if missing_optional:
            self.logger.warning(
                f"Missing optional environment variables: {missing_optional}"
            )
            self.logger.warning("Some features may be limited without these keys")

        # Validate configuration
        self._validate_configuration()

        self.logger.info("Environment validation completed")

    def _validate_configuration(self):
        """Validate configuration settings."""
        try:
            # Check embedding configuration
            embedding_config = self.config.get("embedding", {})
            if not embedding_config.get("model"):
                self.logger.warning("Embedding model not specified, using default")

            # Check vector database configuration
            vector_db_config = self.config.get("vector_db", {})
            if not vector_db_config.get("provider"):
                self.logger.warning(
                    "Vector database provider not specified, using default"
                )

            # Check RAG configuration
            rag_config = self.config.get("rag", {})
            if rag_config.get("top_k", 5) <= 0:
                self.logger.warning("Invalid top_k value, using default")

            self.logger.info("Configuration validation completed")

        except Exception as e:
            self.logger.warning(f"Configuration validation warning: {e}")

    def _initialize_components(self):
        """Initialize all system components with error handling."""
        try:
            self.logger.info("Initializing system components...")

            # Document processing components
            self.logger.info(" Initializing document processing components...")
            self.document_processor = DocumentProcessor(
                self.config_manager.get_section("document_processing")
            )

            self.url_processor = URLProcessor(
                self.config_manager.get_section("url_processing")
            )

            self.text_extractor = TextExtractor(
                self.config_manager.get_section("document_processing")
            )

            # Embedding and storage components
            self.logger.info("Initializing embedding and storage components...")
            embedding_config = self.config_manager.get_section("embedding")
            embedding_config["api_key"] = os.getenv("GEMINI_API_KEY")

            self.embedding_generator = EmbeddingGenerator(embedding_config)

            vector_db_config = self.config_manager.get_section("vector_db")
            vector_db_config["api_key"] = os.getenv("PINECONE_API_KEY")

            self.vector_db = VectorDB(vector_db_config)

            # RAG components
            self.logger.info("Initializing RAG components...")
            self.query_processor = OptimizedQueryProcessor(
                self.embedding_generator,
                self.vector_db,
                self.config_manager.get_section("rag"),
            )

            rag_config = self.config_manager.get_section("rag")
            # Add API keys to RAG config for LLM initialization
            rag_config["gemini_api_key"] = os.getenv("GEMINI_API_KEY")
            rag_config["openai_api_key"] = os.getenv("OPENAI_API_KEY")

            self.response_generator = ResponseGenerator(rag_config)

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f" Failed to initialize components: {str(e)}")
            # Don't raise in demo mode, continue with limited functionality
            self.logger.warning("Some components may not be fully functional")

    def _run_startup_health_checks(self):
        """Run health checks on all components."""
        self.logger.info("Running startup health checks...")

        health_status = {
            "document_processor": False,
            "url_processor": False,
            "text_extractor": False,
            "embedding_generator": False,
            "vector_db": False,
            "query_processor": False,
            "response_generator": False,
        }

        # Check each component
        try:
            if hasattr(self, "document_processor"):
                health_status["document_processor"] = True
                self.logger.info("Document processor: Healthy")
        except:
            self.logger.warning("Document processor: Not available")

        try:
            if hasattr(self, "url_processor"):
                health_status["url_processor"] = True
                self.logger.info("URL processor: Healthy")
        except:
            self.logger.warning("URL processor: Not available")

        try:
            if hasattr(self, "text_extractor"):
                health_status["text_extractor"] = True
                self.logger.info("Text extractor: Healthy")
        except:
            self.logger.warning("Text extractor: Not available")

        try:
            if hasattr(self, "embedding_generator"):
                health_status["embedding_generator"] = True
                self.logger.info("Embedding generator: Healthy")
        except:
            self.logger.warning("Embedding generator: Not available")

        try:
            if hasattr(self, "vector_db"):
                health_status["vector_db"] = True
                self.logger.info("Vector database: Healthy")
        except:
            self.logger.warning("Vector database: Not available")

        try:
            if hasattr(self, "query_processor"):
                health_status["query_processor"] = True
                self.logger.info("Query processor: Healthy")
        except:
            self.logger.warning("Query processor: Not available")

        try:
            if hasattr(self, "response_generator"):
                health_status["response_generator"] = True
                self.logger.info("Response generator: Healthy")
        except:
            self.logger.warning("Response generator: Not available")

        # Overall health
        healthy_components = sum(health_status.values())
        total_components = len(health_status)

        self.logger.info(
            f"Health check complete: {healthy_components}/{total_components} components healthy"
        )

        if healthy_components < total_components:
            self.logger.warning("Some components are not fully functional")
            self.logger.warning("The system will run with limited capabilities")

    def process_document(self, file_path: str) -> dict:
        """
        Process a document through the complete pipeline.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f" Processing document: {file_path}")

            # Check if components are available
            if not all(
                hasattr(self, attr)
                for attr in [
                    "document_processor",
                    "text_extractor",
                    "embedding_generator",
                    "vector_db",
                ]
            ):
                return {
                    "status": "error",
                    "error": "Required components not available",
                    "chunks_processed": 0,
                }

            # Step 1: Extract content from document
            doc_result = self.document_processor.process_document(file_path)

            if not doc_result or "content" not in doc_result:
                return {
                    "status": "error",
                    "error": "Failed to extract content from document",
                    "chunks_processed": 0,
                }

            # Step 2: Extract and chunk text
            text_chunks = self.text_extractor.process_text(
                doc_result["content"], doc_result.get("metadata", {})
            )

            if not text_chunks:
                return {
                    "status": "error",
                    "error": "No text chunks generated",
                    "chunks_processed": 0,
                }

            # Step 3: Generate embeddings
            embedded_chunks = self.embedding_generator.generate_embeddings(text_chunks)

            if not embedded_chunks:
                return {
                    "status": "error",
                    "error": "Failed to generate embeddings",
                    "chunks_processed": len(text_chunks),
                }

            # Step 4: Store in vector database
            storage_success = self.vector_db.store_embeddings(embedded_chunks)

            return {
                "status": "success" if storage_success else "partial_success",
                "chunks_processed": len(text_chunks),
                "chunks_stored": len(embedded_chunks) if storage_success else 0,
                "source": file_path,
            }

        except Exception as e:
            self.logger.error(f" Error processing document: {str(e)}")
            error_info = self.error_handler.handle_error(e, {"file_path": file_path})
            return {
                "status": "error",
                "error": str(e),
                "error_info": error_info,
                "chunks_processed": 0,
            }

    def process_url(self, url: str) -> dict:
        """
        Process a URL through the complete pipeline.

        Args:
            url: URL to process

        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f"Processing URL: {url}")

            # Check if components are available
            if not all(
                hasattr(self, attr)
                for attr in [
                    "url_processor",
                    "text_extractor",
                    "embedding_generator",
                    "vector_db",
                ]
            ):
                return {
                    "status": "error",
                    "error": "Required components not available",
                    "chunks_processed": 0,
                }

            # Step 1: Extract content from URL
            url_result = self.url_processor.process_url(url)

            if not url_result or "content" not in url_result:
                return {
                    "status": "error",
                    "error": "Failed to extract content from URL",
                    "chunks_processed": 0,
                }

            # Step 2: Extract and chunk text
            text_chunks = self.text_extractor.process_text(
                url_result["content"], url_result.get("metadata", {})
            )

            if not text_chunks:
                return {
                    "status": "error",
                    "error": "No text chunks generated",
                    "chunks_processed": 0,
                }

            # Step 3: Generate embeddings
            embedded_chunks = self.embedding_generator.generate_embeddings(text_chunks)

            if not embedded_chunks:
                return {
                    "status": "error",
                    "error": "Failed to generate embeddings",
                    "chunks_processed": len(text_chunks),
                }

            # Step 4: Store in vector database
            storage_success = self.vector_db.store_embeddings(embedded_chunks)

            # Process linked documents if any
            linked_processed = 0
            for linked_doc in url_result.get("linked_documents", []):
                if linked_doc and "content" in linked_doc:
                    try:
                        linked_chunks = self.text_extractor.process_text(
                            linked_doc["content"], linked_doc.get("metadata", {})
                        )
                        if linked_chunks:
                            linked_embedded = (
                                self.embedding_generator.generate_embeddings(
                                    linked_chunks
                                )
                            )
                            if linked_embedded and self.vector_db.store_embeddings(
                                linked_embedded
                            ):
                                linked_processed += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to process linked document: {e}")

            return {
                "status": "success" if storage_success else "partial_success",
                "chunks_processed": len(text_chunks),
                "chunks_stored": len(embedded_chunks) if storage_success else 0,
                "linked_documents_processed": linked_processed,
                "source": url,
            }

        except Exception as e:
            self.logger.error(f" Error processing URL: {str(e)}")
            error_info = self.error_handler.handle_error(e, {"url": url})
            return {
                "status": "error",
                "error": str(e),
                "error_info": error_info,
                "chunks_processed": 0,
            }

    def query(self, question: str) -> dict:
        """
        Process a query and generate a response.

        Args:
            question: User question

        Returns:
            Dictionary with response and metadata
        """
        try:
            self.logger.info(f"Processing query: {question[:100]}...")

            # Check if components are available
            if not all(
                hasattr(self, attr)
                for attr in ["query_processor", "response_generator"]
            ):
                return {
                    "query": question,
                    "response": "Query processing components not available. Please check system configuration.",
                    "sources": [],
                    "confidence": 0.0,
                    "error": "Components not available",
                }

            # Step 1: Process query and retrieve context
            query_result = self.query_processor.process_query(question)

            if query_result.get("error"):
                return {
                    "query": question,
                    "response": f"Query processing failed: {query_result['error']}",
                    "sources": [],
                    "confidence": 0.0,
                    "error": query_result["error"],
                }

            # Step 2: Generate response
            response_result = self.response_generator.generate_response(
                question, query_result.get("context", [])
            )

            # Combine results
            return {
                "query": question,
                "response": response_result.get("response", "No response generated"),
                "sources": response_result.get("sources", []),
                "confidence": response_result.get("confidence", 0.0),
                "context_items": query_result.get("total_results", 0),
                "processing_time": query_result.get("processing_time", 0),
                "generation_time": response_result.get("generation_time", 0),
                "model_used": response_result.get("model_used", "unknown"),
            }

        except Exception as e:
            self.logger.error(f" Error processing query: {str(e)}")
            error_info = self.error_handler.handle_error(e, {"query": question})
            return {
                "query": question,
                "response": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "confidence": 0.0,
                "error": str(e),
                "error_info": error_info,
            }

    def get_system_status(self) -> dict:
        """
        Get comprehensive system status.

        Returns:
            Dictionary with system status information
        """
        try:
            status = {
                "overall_status": "healthy",
                "components": {},
                "configuration": {},
                "environment": {},
            }

            # Check component status
            components = [
                "document_processor",
                "url_processor",
                "text_extractor",
                "embedding_generator",
                "vector_db",
                "query_processor",
                "response_generator",
            ]

            for component in components:
                status["components"][component] = hasattr(self, component)

            # Configuration info
            status["configuration"] = {
                "embedding_model": self.config.get("embedding", {}).get(
                    "model", "unknown"
                ),
                "vector_db_provider": self.config.get("vector_db", {}).get(
                    "provider", "unknown"
                ),
                "rag_top_k": self.config.get("rag", {}).get("top_k", 5),
            }

            # Environment info
            status["environment"] = {
                "gemini_api_available": bool(os.getenv("GEMINI_API_KEY")),
                "pinecone_api_available": bool(os.getenv("PINECONE_API_KEY")),
                "openai_api_available": bool(os.getenv("OPENAI_API_KEY")),
            }

            # Overall status
            healthy_components = sum(status["components"].values())
            total_components = len(status["components"])

            if healthy_components < total_components * 0.8:
                status["overall_status"] = "degraded"
            elif healthy_components < total_components * 0.5:
                status["overall_status"] = "unhealthy"

            return status

        except Exception as e:
            self.logger.error(f" Error getting system status: {e}")
            return {"overall_status": "error", "error": str(e)}


def create_app():
    """
    Create and configure the RAG application.

    Returns:
        Tuple of (RAG system instance, Gradio app instance)
    """
    try:
        # Initialize the RAG system
        rag_system = RAGSystem()

        # Create Gradio interface
        ui_config = rag_system.config_manager.get_section("ui")
        gradio_app = GradioApp(rag_system, ui_config)

        return rag_system, gradio_app

    except Exception as e:
        print(f" Failed to create application: {str(e)}")
        # Create a minimal system for demo purposes
        print("Creating minimal demo system...")

        # Create minimal config
        minimal_config = {
            "ui": {
                "title": "AI Embedded Knowledge Agent (Demo Mode)",
                "description": "Demo mode - some features may be limited. Please configure API keys for full functionality.",
            }
        }

        # Create minimal RAG system
        class MinimalRAGSystem:
            def __init__(self):
                self.config_manager = type(
                    "ConfigManager",
                    (),
                    {
                        "get_section": lambda self, section: minimal_config.get(
                            section, {}
                        )
                    },
                )()

            def process_document(self, file_path):
                return {
                    "status": "error",
                    "error": "Demo mode - document processing not available",
                }

            def process_url(self, url):
                return {
                    "status": "error",
                    "error": "Demo mode - URL processing not available",
                }

            def query(self, question):
                return {
                    "query": question,
                    "response": "Demo mode: Please configure your API keys (GEMINI_API_KEY, PINECONE_API_KEY) to enable full functionality.",
                    "sources": [],
                    "confidence": 0.0,
                }

        rag_system = MinimalRAGSystem()
        gradio_app = GradioApp(rag_system, minimal_config.get("ui", {}))

        return rag_system, gradio_app


def main():
    """Main function to run the application."""
    try:
        print("Starting AI Embedded Knowledge Agent...")
        print("=" * 50)

        # Create the application
        rag_system, gradio_app = create_app()

        # Get launch configuration
        try:
            ui_config = rag_system.config_manager.get_section("ui")
        except:
            ui_config = {}

        # Launch the Gradio interface
        launch_config = {
            "server_name": ui_config.get("server_name", "0.0.0.0"),
            "server_port": ui_config.get("port", 7860),
            "share": ui_config.get("share", False),
            "show_error": True,
            "quiet": False,
        }

        print(
            f"Launching interface on {launch_config['server_name']}:{launch_config['server_port']}"
        )
        print("=" * 50)

        gradio_app.launch(**launch_config)

    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f" Failed to start application: {str(e)}")
        print("Please check your configuration and API keys.")
        sys.exit(1)


if __name__ == "__main__":
    main()
