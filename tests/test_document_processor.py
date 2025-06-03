"""
Unit tests for the DocumentProcessor module.
"""

import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the module to test
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion.document_processor import DocumentProcessor


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = {
            "max_file_size_mb": 50,
            "supported_formats": [".pdf", ".docx", ".csv", ".xlsx", ".pptx"],
        }
        self.processor = DocumentProcessor(self.config)

    def test_init_with_config(self):
        """Test DocumentProcessor initialization with config."""
        processor = DocumentProcessor(self.config)
        self.assertEqual(processor.config, self.config)

    def test_init_without_config(self):
        """Test DocumentProcessor initialization without config."""
        processor = DocumentProcessor()
        self.assertEqual(processor.config, {})

    def test_process_document_file_not_found(self):
        """Test processing a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.processor.process_document("non_existent_file.pdf")

    def test_process_document_unsupported_format(self):
        """Test processing an unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"Test content")
            temp_file_path = temp_file.name

        try:
            with self.assertRaises(ValueError) as context:
                self.processor.process_document(temp_file_path)
            self.assertIn("Unsupported file format", str(context.exception))
        finally:
            os.unlink(temp_file_path)

    def test_process_pdf_document(self):
        """Test processing a PDF document."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"Mock PDF content")
            temp_file_path = temp_file.name

        try:
            result = self.processor.process_document(temp_file_path)

            # Check result structure
            self.assertIn("content", result)
            self.assertIn("metadata", result)
            self.assertIn("source", result)
            self.assertEqual(result["source"], temp_file_path)
        finally:
            os.unlink(temp_file_path)

    def test_process_docx_document(self):
        """Test processing a DOCX document."""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_file:
            temp_file.write(b"Mock DOCX content")
            temp_file_path = temp_file.name

        try:
            result = self.processor.process_document(temp_file_path)

            # Check result structure
            self.assertIn("content", result)
            self.assertIn("metadata", result)
            self.assertIn("source", result)
            self.assertEqual(result["source"], temp_file_path)
        finally:
            os.unlink(temp_file_path)

    def test_process_csv_document(self):
        """Test processing a CSV document."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_file.write(b"col1,col2\nvalue1,value2")
            temp_file_path = temp_file.name

        try:
            result = self.processor.process_document(temp_file_path)

            # Check result structure
            self.assertIn("content", result)
            self.assertIn("metadata", result)
            self.assertIn("source", result)
            self.assertEqual(result["source"], temp_file_path)
        finally:
            os.unlink(temp_file_path)

    def test_process_pptx_document(self):
        """Test processing a PPTX document."""
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as temp_file:
            temp_file.write(b"Mock PPTX content")
            temp_file_path = temp_file.name

        try:
            result = self.processor.process_document(temp_file_path)

            # Check result structure
            self.assertIn("content", result)
            self.assertIn("metadata", result)
            self.assertIn("source", result)
            self.assertEqual(result["source"], temp_file_path)
        finally:
            os.unlink(temp_file_path)

    @patch("ingestion.document_processor.logging")
    def test_logging_on_processing(self, mock_logging):
        """Test that logging occurs during document processing."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"Mock PDF content")
            temp_file_path = temp_file.name

        try:
            self.processor.process_document(temp_file_path)
            # Verify that logger was called (through getLogger)
            mock_logging.getLogger.assert_called()
        finally:
            os.unlink(temp_file_path)

    def test_private_methods_exist(self):
        """Test that private processing methods exist."""
        # Check that private methods are defined
        self.assertTrue(hasattr(self.processor, "_process_pdf"))
        self.assertTrue(hasattr(self.processor, "_process_docx"))
        self.assertTrue(hasattr(self.processor, "_process_spreadsheet"))
        self.assertTrue(hasattr(self.processor, "_process_pptx"))

    def test_private_method_return_structure(self):
        """Test that private methods return correct structure."""
        # Test _process_pdf
        result = self.processor._process_pdf("dummy_path.pdf")
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        self.assertIn("source", result)

        # Test _process_docx
        result = self.processor._process_docx("dummy_path.docx")
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        self.assertIn("source", result)


if __name__ == "__main__":
    unittest.main()
