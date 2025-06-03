"""
Unit tests for the URLProcessor module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Import the module to test
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion.url_processor import URLProcessor


class TestURLProcessor(unittest.TestCase):
    """Test cases for URLProcessor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = {
            "max_depth": 2,
            "follow_links": True,
            "max_pages": 5,
            "timeout": 10,
        }
        self.processor = URLProcessor(self.config)

    def test_init_with_config(self):
        """Test URLProcessor initialization with config."""
        processor = URLProcessor(self.config)
        self.assertEqual(processor.config, self.config)
        self.assertEqual(processor.max_depth, 2)
        self.assertEqual(processor.follow_links, True)
        self.assertEqual(processor.max_pages, 5)
        self.assertEqual(processor.timeout, 10)

    def test_init_without_config(self):
        """Test URLProcessor initialization without config."""
        processor = URLProcessor()
        self.assertEqual(processor.config, {})
        # Check default values
        self.assertEqual(processor.max_depth, 1)
        self.assertEqual(processor.follow_links, True)
        self.assertEqual(processor.max_pages, 10)
        self.assertEqual(processor.timeout, 10)

    def test_process_url_empty_url(self):
        """Test processing an empty URL."""
        result = self.processor.process_url("")
        self.assertEqual(result, {})

    def test_process_url_none_url(self):
        """Test processing a None URL."""
        result = self.processor.process_url(None)
        self.assertEqual(result, {})

    def test_process_url_max_depth_exceeded(self):
        """Test processing URL when max depth is exceeded."""
        result = self.processor.process_url("https://example.com", depth=10)
        self.assertEqual(result, {})

    def test_process_url_max_pages_exceeded(self):
        """Test processing URL when max pages limit is reached."""
        # Fill up the visited URLs to max capacity
        for i in range(self.processor.max_pages):
            self.processor.visited_urls.add(f"https://example{i}.com")

        result = self.processor.process_url("https://newexample.com")
        self.assertEqual(result, {})

    def test_process_url_already_visited(self):
        """Test processing a URL that has already been visited."""
        url = "https://example.com"
        self.processor.visited_urls.add(url)

        result = self.processor.process_url(url)
        self.assertEqual(result, {})

    @patch("ingestion.url_processor.URLProcessor._extract_content")
    @patch("ingestion.url_processor.URLProcessor._extract_links")
    def test_process_url_success(self, mock_extract_links, mock_extract_content):
        """Test successful URL processing."""
        url = "https://example.com"
        mock_content = "Test content"
        mock_metadata = {"title": "Test Page", "url": url}
        mock_links = ["https://example.com/page1", "https://example.com/page2"]

        mock_extract_content.return_value = (mock_content, mock_metadata)
        mock_extract_links.return_value = mock_links

        # Configure processor to not follow links for this test
        self.processor.follow_links = False

        result = self.processor.process_url(url)

        # Check result structure
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        self.assertIn("source", result)
        self.assertIn("linked_documents", result)

        self.assertEqual(result["content"], mock_content)
        self.assertEqual(result["metadata"], mock_metadata)
        self.assertEqual(result["source"], url)
        self.assertEqual(result["linked_documents"], [])

        # Verify URL was added to visited set
        self.assertIn(url, self.processor.visited_urls)

    @patch("ingestion.url_processor.URLProcessor._extract_content")
    def test_process_url_with_exception(self, mock_extract_content):
        """Test URL processing when an exception occurs."""
        url = "https://example.com"
        mock_extract_content.side_effect = Exception("Network error")

        result = self.processor.process_url(url)
        self.assertEqual(result, {})

    def test_extract_content_placeholder(self):
        """Test the _extract_content method placeholder implementation."""
        url = "https://example.com"
        content, metadata = self.processor._extract_content(url)

        # Check that placeholder content is returned
        self.assertIsInstance(content, str)
        self.assertIsInstance(metadata, dict)
        self.assertIn("url", metadata)
        self.assertEqual(metadata["url"], url)

    def test_extract_links_placeholder(self):
        """Test the _extract_links method placeholder implementation."""
        url = "https://example.com"
        links = self.processor._extract_links(url)

        # Check that empty list is returned (placeholder implementation)
        self.assertIsInstance(links, list)
        self.assertEqual(len(links), 0)

    def test_reset_method(self):
        """Test the reset method."""
        # Add some URLs to visited set
        self.processor.visited_urls.add("https://example1.com")
        self.processor.visited_urls.add("https://example2.com")

        # Reset the processor
        self.processor.reset()

        # Check that visited URLs are cleared
        self.assertEqual(len(self.processor.visited_urls), 0)

    @patch("ingestion.url_processor.logging")
    def test_logging_on_processing(self, mock_logging):
        """Test that logging occurs during URL processing."""
        url = "https://example.com"
        self.processor.process_url(url)

        # Verify that logger was called (through getLogger)
        mock_logging.getLogger.assert_called()

    def test_private_methods_exist(self):
        """Test that private processing methods exist."""
        # Check that private methods are defined
        self.assertTrue(hasattr(self.processor, "_extract_content"))
        self.assertTrue(hasattr(self.processor, "_extract_links"))

    @patch("ingestion.url_processor.URLProcessor._extract_content")
    @patch("ingestion.url_processor.URLProcessor._extract_links")
    def test_process_url_with_links_following(
        self, mock_extract_links, mock_extract_content
    ):
        """Test URL processing with link following enabled."""
        url = "https://example.com"
        mock_content = "Test content"
        mock_metadata = {"title": "Test Page", "url": url}
        mock_links = ["https://example.com/page1"]

        mock_extract_content.return_value = (mock_content, mock_metadata)
        mock_extract_links.return_value = mock_links

        # Enable link following
        self.processor.follow_links = True
        self.processor.max_depth = 2

        result = self.processor.process_url(url, depth=0)

        # Check that the method was called for both original URL and linked URL
        self.assertEqual(mock_extract_content.call_count, 2)  # Original + 1 linked URL

        # Check result structure
        self.assertIn("linked_documents", result)
        self.assertEqual(len(result["linked_documents"]), 1)


if __name__ == "__main__":
    unittest.main()
