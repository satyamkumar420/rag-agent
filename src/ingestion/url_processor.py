"""
URL Processor Module

This module is responsible for crawling and extracting content from provided URLs,
including nested documents and links with complete web scraping functionality.

Technologies: BeautifulSoup, requests, trafilatura
"""

import logging
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from urllib.parse import urlparse, urljoin, urlunparse
from urllib.robotparser import RobotFileParser

# Import web scraping libraries
try:
    import requests
    from bs4 import BeautifulSoup
    import trafilatura
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as e:
    logging.warning(f"Some web scraping libraries are not installed: {e}")

from utils.error_handler import URLProcessingError, error_handler, ErrorType


class URLProcessor:
    """
    Processes URLs to extract content from web pages and linked documents with full functionality.

    Features:
    - Web page content extraction with trafilatura
    - Recursive link following with depth control
    - Rate limiting and retry logic
    - Robots.txt respect
    - Multiple content type handling
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the URLProcessor with configuration.

        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.max_depth = self.config.get("max_depth", 1)
        self.follow_links = self.config.get("follow_links", True)
        self.max_pages = self.config.get("max_pages", 10)
        self.timeout = self.config.get("timeout", 10)
        self.user_agent = self.config.get("user_agent", "RAG-AI-Bot/1.0")
        self.respect_robots_txt = self.config.get("respect_robots_txt", True)
        self.rate_limit_delay = self.config.get("rate_limit_delay", 1.0)

        # Retry configuration
        self.max_retries = 3
        self.backoff_factor = 0.3

        # Track visited URLs and robots.txt cache
        self.visited_urls: Set[str] = set()
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.last_request_time: Dict[str, float] = {}

        # Setup session with retry strategy
        self.session = self._setup_session()

    def _setup_session(self) -> requests.Session:
        """
        Setup requests session with retry strategy and headers.

        Returns:
            Configured requests session
        """
        session = requests.Session()

        # Retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # 🏷Default headers
        session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )

        return session

    @error_handler(ErrorType.URL_PROCESSING)
    def process_url(self, url: str, depth: int = 0) -> Dict[str, Any]:
        """
        Process a URL and extract its content with full functionality.

        Args:
            url: The URL to process
            depth: Current crawling depth

        Returns:
            Dictionary containing extracted text and metadata
        """
        # Validation checks
        if not url or not self._is_valid_url(url):
            raise URLProcessingError(f"Invalid URL: {url}", url)

        if depth > self.max_depth:
            self.logger.info(f"🛑 Max depth reached for: {url}")
            return {}

        if len(self.visited_urls) >= self.max_pages:
            self.logger.info(f"🛑 Max pages limit reached")
            return {}

        if url in self.visited_urls:
            self.logger.info(f"Already visited: {url}")
            return {}

        # Check robots.txt if enabled
        if self.respect_robots_txt and not self._can_fetch(url):
            self.logger.info(f"Robots.txt disallows: {url}")
            return {}

        self.visited_urls.add(url)
        self.logger.info(f"Processing URL: {url} (depth: {depth})")

        try:
            # Rate limiting
            self._apply_rate_limit(url)

            # Fetch and extract content
            content, metadata = self._extract_content(url)

            if not content:
                self.logger.warning(f"No content extracted from: {url}")
                return {}

            result = {
                "content": content,
                "metadata": metadata,
                "source": url,
                "depth": depth,
                "linked_documents": [],
                "document_type": "webpage",
                "crawl_stats": {
                    "max_depth_configured": self.max_depth,
                    "follow_links_enabled": self.follow_links,
                    "current_depth": depth,
                },
            }

            #  Follow links if configured and not at max depth
            if (
                self.follow_links
                and depth < self.max_depth
                and len(self.visited_urls) < self.max_pages
            ):
                links = self._extract_links(url, content)
                self.logger.info(f" Found {len(links)} links on {url}")

                for link in links[:5]:  # Limit links per page
                    try:
                        linked_content = self.process_url(link, depth + 1)
                        if linked_content:
                            result["linked_documents"].append(linked_content)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to process linked URL {link}: {str(e)}"
                        )
                        continue

            return result

        except Exception as e:
            raise URLProcessingError(f"Error processing URL: {str(e)}", url)

    def process_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple URLs in batch.

        Args:
            urls: List of URLs to process

        Returns:
            List of processed URL results
        """
        results = []
        self.logger.info(f"Processing batch of {len(urls)} URLs")

        for i, url in enumerate(urls):
            try:
                result = self.process_url(url)
                if result:
                    results.append(result)
                self.logger.info(f"Processed {i+1}/{len(urls)}: {url}")
            except Exception as e:
                self.logger.error(f"❌ Failed to process {url}: {str(e)}")
                continue

        return results

    def _is_valid_url(self, url: str) -> bool:
        """
        Validate URL format and scheme.

        Args:
            url: URL to validate

        Returns:
            True if URL is valid
        """
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and parsed.scheme in ["http", "https"]
        except Exception:
            return False

    def _can_fetch(self, url: str) -> bool:
        """
        Check if URL can be fetched according to robots.txt.

        Args:
            url: URL to check

        Returns:
            True if URL can be fetched
        """
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            if base_url not in self.robots_cache:
                robots_url = urljoin(base_url, "/robots.txt")
                rp = RobotFileParser()
                rp.set_url(robots_url)

                try:
                    rp.read()
                    self.robots_cache[base_url] = rp
                except Exception:
                    # If robots.txt can't be fetched, assume allowed
                    return True

            return self.robots_cache[base_url].can_fetch(self.user_agent, url)

        except Exception:
            # If robots.txt check fails, assume allowed
            return True

    def _apply_rate_limit(self, url: str) -> None:
        """
        Apply rate limiting between requests to the same domain.

        Args:
            url: URL being processed
        """
        domain = urlparse(url).netloc
        current_time = time.time()

        if domain in self.last_request_time:
            time_since_last = current_time - self.last_request_time[domain]
            if time_since_last < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last
                self.logger.info(
                    f"Rate limiting: sleeping {sleep_time:.1f}s for {domain}"
                )
                time.sleep(sleep_time)

        self.last_request_time[domain] = time.time()

    def _extract_content(self, url: str) -> tuple:
        """
        Extract content from a web page using trafilatura with BeautifulSoup fallback.

        Args:
            url: The URL to extract content from

        Returns:
            Tuple of (content, metadata)
        """
        self.logger.info(f"Extracting content from: {url}")

        try:
            # Fetch the page
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Basic metadata
            metadata = {
                "url": url,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "content_length": len(response.content),
                "extracted_time": datetime.now().isoformat(),
                "encoding": response.encoding or "utf-8",
            }

            # Check content type
            content_type = response.headers.get("content-type", "").lower()

            if "application/pdf" in content_type:
                return self._handle_pdf_url(response, metadata)
            elif "text/html" not in content_type and "text/plain" not in content_type:
                self.logger.warning(f"Unsupported content type: {content_type}")
                return "", metadata

            # Primary method: trafilatura (best for content extraction)
            try:
                content = trafilatura.extract(
                    response.text,
                    include_comments=False,
                    include_tables=True,
                    include_formatting=False,
                    favor_precision=True,
                )

                if content and len(content.strip()) > 50:  # Minimum content threshold
                    # Extract additional metadata with trafilatura
                    metadata_extracted = trafilatura.extract_metadata(response.text)
                    if metadata_extracted:
                        metadata.update(
                            {
                                "title": metadata_extracted.title or "",
                                "author": metadata_extracted.author or "",
                                "description": metadata_extracted.description or "",
                                "sitename": metadata_extracted.sitename or "",
                                "date": metadata_extracted.date or "",
                            }
                        )

                    metadata.update(
                        {
                            "extraction_method": "trafilatura",
                            "word_count": len(content.split()),
                            "character_count": len(content),
                        }
                    )

                    return content.strip(), metadata

            except Exception as e:
                self.logger.warning(f"Trafilatura failed: {str(e)}")

            # Fallback method: BeautifulSoup
            return self._extract_with_beautifulsoup(response.text, metadata)

        except requests.RequestException as e:
            raise URLProcessingError(f"Failed to fetch URL: {str(e)}", url)
        except Exception as e:
            raise URLProcessingError(f"Content extraction failed: {str(e)}", url)

    def _extract_with_beautifulsoup(self, html: str, metadata: Dict[str, Any]) -> tuple:
        """
        Fallback content extraction using BeautifulSoup.

        Args:
            html: HTML content
            metadata: Existing metadata dictionary

        Returns:
            Tuple of (content, metadata)
        """
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Extract metadata
            title_tag = soup.find("title")
            if title_tag:
                metadata["title"] = title_tag.get_text().strip()

            # Meta tags
            for meta in soup.find_all("meta"):
                name = meta.get("name", "").lower()
                content = meta.get("content", "")
                if name == "description":
                    metadata["description"] = content
                elif name == "author":
                    metadata["author"] = content

            # Remove unwanted elements
            for element in soup(
                ["script", "style", "nav", "header", "footer", "aside"]
            ):
                element.decompose()

            # Extract main content
            content_selectors = [
                "main",
                "article",
                ".content",
                "#content",
                ".post",
                ".entry",
            ]

            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator="\n", strip=True)
                    break

            # Fallback to body if no main content found
            if not content:
                body = soup.find("body")
                if body:
                    content = body.get_text(separator="\n", strip=True)

            # Clean and validate content
            content = re.sub(r"\n\s*\n", "\n\n", content)  # Clean multiple newlines
            content = content.strip()

            metadata.update(
                {
                    "extraction_method": "beautifulsoup",
                    "word_count": len(content.split()),
                    "character_count": len(content),
                }
            )

            return content, metadata

        except Exception as e:
            self.logger.error(f"❌ BeautifulSoup extraction failed: {str(e)}")
            return "", metadata

    def _handle_pdf_url(
        self, response: requests.Response, metadata: Dict[str, Any]
    ) -> tuple:
        """
        📄 Handle PDF content from URL.

        Args:
            response: HTTP response containing PDF
            metadata: Existing metadata

        Returns:
            Tuple of (content, metadata)
        """
        self.logger.info("📄 Detected PDF content, extracting text...")

        try:
            # Save PDF temporarily and process with document processor
            import tempfile
            import os
            from .document_processor import DocumentProcessor

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file.flush()

                # Process PDF
                doc_processor = DocumentProcessor(self.config)
                result = doc_processor.process_document(tmp_file.name)

                # Cleanup
                os.unlink(tmp_file.name)

                metadata.update(
                    {
                        "document_type": "pdf_from_url",
                        "extraction_method": "document_processor",
                    }
                )
                metadata.update(result.get("metadata", {}))

                return result.get("content", ""), metadata

        except Exception as e:
            self.logger.error(f"❌ PDF extraction failed: {str(e)}")
            return "", metadata

    def _extract_links(self, url: str, content: str) -> List[str]:
        """
         Extract links from a web page.

        Args:
            url: The source URL
            content: Page content (for context)

        Returns:
            List of discovered URLs
        """
        self.logger.info(f" Extracting links from: {url}")

        try:
            response = self.session.get(url, timeout=self.timeout)
            soup = BeautifulSoup(response.text, "html.parser")

            links = []
            base_domain = urlparse(url).netloc

            for a_tag in soup.find_all("a", href=True):
                href = a_tag.get("href")
                if not href:
                    continue

                #  Convert relative URLs to absolute
                absolute_url = urljoin(url, href)

                # Filter links
                if self._should_follow_link(absolute_url, base_domain):
                    links.append(absolute_url)

            # 🎯 Remove duplicates and limit
            unique_links = list(dict.fromkeys(links))  # Preserve order
            return unique_links[:20]  # Limit to prevent explosion

        except Exception as e:
            self.logger.error(f"❌ Link extraction failed: {str(e)}")
            return []

    def _should_follow_link(self, url: str, base_domain: str) -> bool:
        """
        Determine if a link should be followed.

        Args:
            url: URL to check
            base_domain: Base domain of the source page

        Returns:
            True if link should be followed
        """
        try:
            parsed = urlparse(url)

            # Skip non-HTTP(S) links
            if parsed.scheme not in ["http", "https"]:
                return False

            # Skip already visited
            if url in self.visited_urls:
                return False

            # Skip file downloads (basic check)
            path = parsed.path.lower()
            skip_extensions = [
                ".pdf",
                ".doc",
                ".docx",
                ".zip",
                ".exe",
                ".dmg",
                ".jpg",
                ".png",
                ".gif",
            ]
            if any(path.endswith(ext) for ext in skip_extensions):
                return False

            # Skip fragments and query-heavy URLs
            if parsed.fragment or len(parsed.query) > 100:
                return False

            # Prefer same domain (but allow subdomains)
            link_domain = parsed.netloc
            if not (
                link_domain == base_domain or link_domain.endswith("." + base_domain)
            ):
                return False

            return True

        except Exception:
            return False

    def reset(self):
        """Reset the processor state, clearing visited URLs and caches."""
        self.visited_urls.clear()
        self.robots_cache.clear()
        self.last_request_time.clear()
        self.logger.info("URL processor state reset")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        return {
            "urls_processed": len(self.visited_urls),
            "domains_cached": len(self.robots_cache),
            "rate_limited_domains": len(self.last_request_time),
            "max_pages_limit": self.max_pages,
            "max_depth_limit": self.max_depth,
        }
