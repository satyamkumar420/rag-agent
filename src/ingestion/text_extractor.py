"""
Text Extractor Module

This module is responsible for cleaning, normalizing, and chunking text
from various sources with complete NLP functionality.

Technologies: NLTK, spaCy, regex, langdetect
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import unicodedata

# Import NLP libraries
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import spacy
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException as LangDetectError

    # Download required NLTK data
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

except ImportError as e:
    logging.warning(f"Some NLP libraries are not installed: {e}")

from utils.error_handler import error_handler, ErrorType


class TextExtractor:
    """
    Cleans, normalizes, and chunks text from various sources with intelligent processing.

    Features:
    - Advanced text cleaning and normalization 
    - Language detection 
    - Intelligent sentence segmentation 
    - Smart text chunking with overlap 
    - Metadata preservation 
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TextExtractor with configuration.

        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)
        self.min_chunk_size = self.config.get("min_chunk_size", 100)
        self.max_chunk_size = self.config.get("max_chunk_size", 2000)

        # NLP settings
        self.enable_language_detection = self.config.get(
            "enable_language_detection", True
        )
        self.preserve_formatting = self.config.get("preserve_formatting", False)
        self.remove_stopwords = self.config.get("remove_stopwords", False)

        # Initialize NLP components
        self.nlp = None
        self.stemmer = None
        self.stop_words = set()

        self._initialize_nlp_components()

    def _initialize_nlp_components(self):
        """Initialize NLP components with error handling."""
        try:
            # Load spaCy model for advanced processing
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("spaCy model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load spaCy model: {str(e)}")

        try:
            # Initialize NLTK components
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words("english"))
            self.logger.info("NLTK components initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize NLTK components: {str(e)}")

    @error_handler(ErrorType.DOCUMENT_PROCESSING)
    def process_text(
        self,
        text: Union[str, List[str]],
        metadata: Optional[Dict[str, Any]] = None,
        preserve_structure: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Process text by cleaning, normalizing, and chunking with intelligence.

        Args:
            text: Raw text content (string or list of strings)
            metadata: Optional metadata to include with each chunk
            preserve_structure: Whether to preserve original text structure

        Returns:
            List of dictionaries containing processed text chunks and metadata
        """
        if not text:
            return []

        # Convert list to string if needed
        if isinstance(text, list):
            text = "\n".join(str(item) for item in text if item)

        if not text.strip():
            return []

        self.logger.info(f"Processing text: {len(text)} characters")

        # Detect language
        language = self._detect_language(text)

        # Clean and normalize the text
        cleaned_text = self._clean_text(text, preserve_structure)

        if len(cleaned_text.strip()) < self.min_chunk_size:
            self.logger.warning(
                f"Text too short after cleaning: {len(cleaned_text)} chars"
            )
            return []

        # Split text into chunks
        chunks = self._chunk_text(cleaned_text)

        # Prepare result with enhanced metadata
        result = []
        base_metadata = metadata.copy() if metadata else {}
        base_metadata.update(
            {
                "language": language,
                "original_length": len(text),
                "cleaned_length": len(cleaned_text),
                "chunk_count": len(chunks),
                "processing_time": datetime.now().isoformat(),
                "chunk_size_config": self.chunk_size,
                "chunk_overlap_config": self.chunk_overlap,
            }
        )

        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_stats = self._analyze_chunk(chunk)

            chunk_metadata.update(
                {
                    "chunk_index": i,
                    "chunk_id": f"chunk_{i}_{hash(chunk) % 10000}",
                    **chunk_stats,
                }
            )

            result.append({"content": chunk, "metadata": chunk_metadata})

        self.logger.info(f"Processed text into {len(chunks)} chunks")
        return result

    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the text.

        Args:
            text: Text to analyze

        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        if not self.enable_language_detection:
            return "en"  # Default to English

        try:
            # Use a sample of text for detection (first 1000 chars)
            sample = text[:1000].strip()
            if len(sample) < 50:  # Too short for reliable detection
                return "en"

            language = detect(sample)
            self.logger.info(f"Detected language: {language}")
            return language

        except (LangDetectError, Exception) as e:
            self.logger.warning(f"Language detection failed: {str(e)}")
            return "en"  # Default to English

    def _clean_text(self, text: str, preserve_structure: bool = False) -> str:
        """
        Clean and normalize text with advanced processing.

        Args:
            text: Raw text to clean
            preserve_structure: Whether to preserve formatting

        Returns:
            Cleaned and normalized text
        """
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        if not preserve_structure:
            # Basic cleaning operations
            # Remove excessive whitespace but preserve paragraph breaks
            text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces/tabs to single space
            text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Multiple newlines to double

            # Remove or normalize special characters
            # Keep basic punctuation and common symbols
            text = re.sub(r'[^\w\s.,;:!?\'"\-()[\]{}/@#$%&*+=<>|\\~`\n]', " ", text)

            # Clean up whitespace again
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n\s*\n+", "\n\n", text)

        # Remove common artifacts
        # Remove page numbers and headers/footers patterns
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)  # Standalone page numbers
        text = re.sub(r"\n\s*Page \d+.*?\n", "\n", text, flags=re.IGNORECASE)

        # Remove excessive punctuation
        text = re.sub(r"[.]{3,}", "...", text)  # Multiple dots
        text = re.sub(r"[-]{3,}", "---", text)  # Multiple dashes

        # Final cleanup
        text = text.strip()

        return text

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with intelligent boundary detection.

        Args:
            text: Cleaned text to chunk

        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []

        # Try intelligent chunking with spaCy first
        if self.nlp:
            try:
                return self._chunk_with_spacy(text)
            except Exception as e:
                self.logger.warning(f"spaCy chunking failed: {str(e)}")

        # Fallback to NLTK sentence-based chunking
        try:
            return self._chunk_with_sentences(text)
        except Exception as e:
            self.logger.warning(f"Sentence chunking failed: {str(e)}")

        # Final fallback to character-based chunking
        return self._chunk_by_characters(text)

    def _chunk_with_spacy(self, text: str) -> List[str]:
        """
        Intelligent chunking using spaCy for better semantic boundaries.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_size = len(sent_text)

            # 📏 Check if adding this sentence exceeds chunk size
            if current_size + sent_size > self.chunk_size and current_chunk:
                # 📦 Finalize current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_chunk, overlap_size = self._create_overlap(current_chunk)
                current_chunk = overlap_chunk
                current_size = overlap_size

            current_chunk.append(sent_text)
            current_size += sent_size

        # 📦 Add the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(chunk_text)

        return chunks

    def _chunk_with_sentences(self, text: str) -> List[str]:
        """
        Chunk text using NLTK sentence tokenization.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            sentence_size = len(sentence)

            # 📏 Check chunk size limit
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # 📦 Finalize current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Create overlap
                overlap_chunk, overlap_size = self._create_overlap(current_chunk)
                current_chunk = overlap_chunk
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        # 📦 Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(chunk_text)

        return chunks

    def _chunk_by_characters(self, text: str) -> List[str]:
        """
        Fallback character-based chunking with boundary detection.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to find a good boundary
            if end < len(text):
                # Look for sentence boundaries first
                for boundary in [". ", "! ", "? ", "\n\n", "\n", ". "]:
                    boundary_pos = text.rfind(boundary, start, end)
                    if boundary_pos > start + self.min_chunk_size:
                        end = boundary_pos + len(boundary)
                        break

            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)

            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def _create_overlap(self, sentences: List[str]) -> tuple:
        """
        Create overlap from previous chunk sentences.

        Args:
            sentences: List of sentences from previous chunk

        Returns:
            Tuple of (overlap_sentences, overlap_size)
        """
        overlap_sentences = []
        overlap_size = 0

        # Add sentences from the end for overlap
        for sentence in reversed(sentences):
            if overlap_size + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_size += len(sentence)
            else:
                break

        return overlap_sentences, overlap_size

    def _analyze_chunk(self, chunk: str) -> Dict[str, Any]:
        """
        Analyze chunk statistics and properties.

        Args:
            chunk: Text chunk to analyze

        Returns:
            Dictionary with chunk statistics
        """
        words = chunk.split()

        stats = {
            "character_count": len(chunk),
            "word_count": len(words),
            "sentence_count": len(sent_tokenize(chunk)) if chunk else 0,
            "avg_word_length": (
                sum(len(word) for word in words) / len(words) if words else 0
            ),
        }

        # Advanced analysis with spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(chunk)
                stats.update(
                    {
                        "entity_count": len(doc.ents),
                        "noun_count": len(
                            [token for token in doc if token.pos_ == "NOUN"]
                        ),
                        "verb_count": len(
                            [token for token in doc if token.pos_ == "VERB"]
                        ),
                    }
                )
            except Exception:
                pass  # Skip advanced analysis if it fails

        return stats

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text using NLP techniques.

        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to return

        Returns:
            List of extracted keywords
        """
        if not self.nlp:
            return []

        try:
            doc = self.nlp(text)

            # Extract keywords based on POS tags and frequency
            keywords = []
            word_freq = {}

            for token in doc:
                if (
                    token.pos_ in ["NOUN", "PROPN", "ADJ"]
                    and not token.is_stop
                    and not token.is_punct
                    and len(token.text) > 2
                ):

                    word = token.lemma_.lower()
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Sort by frequency and return top keywords
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, freq in sorted_words[:max_keywords]]

            return keywords

        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {str(e)}")
            return []

    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get comprehensive text statistics.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with text statistics
        """
        words = text.split()
        sentences = sent_tokenize(text) if text else []

        stats = {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len([p for p in text.split("\n\n") if p.strip()]),
            "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "avg_chars_per_word": (
                sum(len(word) for word in words) / len(words) if words else 0
            ),
            "language": self._detect_language(text),
        }

        return stats
