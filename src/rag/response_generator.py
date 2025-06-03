"""
Response Generator Module

This module is responsible for generating coherent responses based on
retrieved knowledge using LangChain RAG.

Technology: LangChain RAG (Retrieval Augmented Generation)
"""

import logging
import time
import os
from typing import Dict, List, Any, Optional
from datetime import datetime


class ResponseGenerator:
    """
    Generates coherent responses based on retrieved knowledge.

    Features:
    - Context-aware response generation
    - Source attribution and confidence scoring
    - Multiple LLM provider support (Gemini, OpenAI)
    - Response quality assessment
    - Template-based fallback generation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ResponseGenerator with configuration.

        Args:
            config: Configuration dictionary with generation parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.max_tokens = self.config.get("max_tokens", 500)
        self.temperature = self.config.get("temperature", 0.7)
        self.include_sources = self.config.get("include_sources", True)

        # Initialize LLM providers
        self.llm = None
        self.gemini_client = None
        self.openai_client = None

        self._initialize_llm_providers()

        # Response templates
        self.response_templates = {
            "no_context": "I don't have enough information to answer your question. Please try uploading relevant documents or providing URLs.",
            "error": "I encountered an error while generating the response. Please try again.",
            "insufficient_confidence": "Based on the available information, I found some relevant content, but I'm not confident enough to provide a definitive answer.",
        }

        self.logger.info("ResponseGenerator initialized with advanced features")

    def _initialize_llm_providers(self):
        """Initialize available LLM providers."""
        try:
            # Try to initialize Gemini
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if gemini_api_key:
                try:
                    import google.generativeai as genai

                    genai.configure(api_key=gemini_api_key)
                    self.gemini_client = genai.GenerativeModel(
                        "gemini-2.5-flash-preview-05-20"
                    )
                    self.logger.info("Gemini client initialized")
                except ImportError:
                    self.logger.warning("Gemini SDK not available")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Gemini: {e}")

            # Try to initialize OpenAI
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                try:
                    import openai

                    self.openai_client = openai.OpenAI(api_key=openai_api_key)
                    self.logger.info("OpenAI client initialized")
                except ImportError:
                    self.logger.warning("OpenAI SDK not available")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize OpenAI: {e}")

            # Try to initialize LangChain
            try:
                if self.gemini_client:
                    from langchain_google_genai import ChatGoogleGenerativeAI

                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash-preview-05-20",
                        temperature=self.temperature,
                        google_api_key=gemini_api_key,
                    )
                elif self.openai_client:
                    from langchain_openai import ChatOpenAI

                    self.llm = ChatOpenAI(
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        openai_api_key=openai_api_key,
                    )

                if self.llm:
                    self.logger.info("LangChain LLM initialized")

            except ImportError:
                self.logger.warning("LangChain not available")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LangChain: {e}")

        except Exception as e:
            self.logger.error(f"❌ Error initializing LLM providers: {e}")

    def generate_response(
        self, query: str, context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a response based on the query and retrieved context.

        Args:
            query: Original user query
            context: List of retrieved context items with text and metadata

        Returns:
            Dictionary containing the generated response and metadata
        """
        if not query:
            return {
                "response": "I need a question to answer.",
                "sources": [],
                "confidence": 0.0,
                "error": "No query provided",
            }

        if not context:
            return {
                "response": self.response_templates["no_context"],
                "sources": [],
                "confidence": 0.0,
                "error": "No context available",
            }

        self.logger.info(f"Generating response for query: {query[:100]}...")
        start_time = time.time()

        try:
            # Prepare context for generation
            formatted_context = self._format_context(context)

            # Calculate initial confidence based on context quality
            base_confidence = self._calculate_confidence(context)

            # Generate response using available LLM
            response_result = self._generate_with_llm(query, formatted_context)

            if not response_result["success"]:
                # Fallback to template-based generation
                response_result = self._fallback_generation(query, formatted_context)

            # Extract sources from context
            sources = self._extract_sources(context) if self.include_sources else []

            # Assess response quality
            quality_score = self._assess_response_quality(
                response_result["response"], query, context
            )

            # Calculate final confidence
            final_confidence = min(base_confidence * quality_score, 1.0)

            # Check if confidence is too low
            if final_confidence < 0.3:
                response_text = self.response_templates["insufficient_confidence"]
                final_confidence = 0.2
            else:
                response_text = response_result["response"]

            result = {
                "response": response_text,
                "sources": sources,
                "confidence": final_confidence,
                "context_items": len(context),
                "generation_time": time.time() - start_time,
                "model_used": response_result.get("model", "fallback"),
                "quality_score": quality_score,
            }

            self.logger.info(f"Response generated in {result['generation_time']:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Error generating response: {str(e)}")
            return {
                "response": self.response_templates["error"],
                "sources": [],
                "confidence": 0.0,
                "error": str(e),
                "generation_time": time.time() - start_time,
            }

    def _generate_with_llm(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate response using available LLM providers.

        Args:
            query: User query
            context: Formatted context string

        Returns:
            Dictionary with generation result
        """
        # Create RAG prompt
        prompt = self._create_rag_prompt(query, context)

        # Try LangChain first
        if self.llm:
            try:
                from langchain.schema import HumanMessage

                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                return {
                    "success": True,
                    "response": response.content,
                    "model": "langchain",
                }
            except Exception as e:
                self.logger.warning(f"LangChain generation failed: {e}")

        # Try Gemini directly
        if self.gemini_client:
            try:
                response = self.gemini_client.generate_content(prompt)
                return {
                    "success": True,
                    "response": response.text,
                    "model": "gemini-2.5-flash-preview-05-20",
                }
            except Exception as e:
                self.logger.warning(f"Gemini generation failed: {e}")

        # Try OpenAI directly
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return {
                    "success": True,
                    "response": response.choices[0].message.content,
                    "model": self.model,
                }
            except Exception as e:
                self.logger.warning(f"OpenAI generation failed: {e}")

        return {"success": False, "response": "", "model": "none"}

    def _create_rag_prompt(self, query: str, context: str) -> str:
        """
        Create an enhanced prompt template for RAG generation.

        Args:
            query: User query
            context: Formatted context

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an AI assistant that answers questions based on provided context. Follow these guidelines:

1. Answer the question using ONLY the information provided in the context
2. If the context doesn't contain enough information, clearly state this
3. Cite specific sources when making claims
4. Be concise but comprehensive
5. If multiple sources provide different information, acknowledge this
6. Use a professional and helpful tone

Context Information:
{context}

Question: {query}

Instructions:
- Provide a clear, well-structured answer
- Include relevant details from the context
- If uncertain, express the level of confidence
- Do not make up information not present in the context

Answer:"""

        return prompt

    def _fallback_generation(self, query: str, context: str) -> Dict[str, Any]:
        """
        Fallback response generation when LLM is not available.

        Args:
            query: User query
            context: Formatted context

        Returns:
            Dictionary with generation result
        """
        self.logger.info("Using fallback generation")

        # Extract key information from context
        context_lines = context.split("\n")
        relevant_lines = [
            line.strip()
            for line in context_lines
            if line.strip() and not line.startswith("[Source:")
        ]

        if not relevant_lines:
            return {
                "success": True,
                "response": self.response_templates["no_context"],
                "model": "fallback",
            }

        # Create a structured response
        response_parts = [
            f"Based on the available information regarding '{query}':",
            "",
        ]

        # Add key information
        for i, line in enumerate(relevant_lines[:3]):  # Limit to 3 most relevant
            if len(line) > 50:  # Only include substantial content
                response_parts.append(f"• {line}")

        response_parts.extend(
            [
                "",
                "Note: This response is generated using available context. For more detailed analysis, please ensure proper language model integration.",
            ]
        )

        response = "\n".join(response_parts)

        return {
            "success": True,
            "response": response,
            "model": "fallback",
        }

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        Format the retrieved context for use in response generation.

        Args:
            context: List of context items

        Returns:
            Formatted context string
        """
        formatted_parts = []

        for i, item in enumerate(context):
            text = item.get("text", "")
            source = item.get("source", f"Source {i+1}")
            score = item.get("score", 0.0)

            # Format each context item with metadata
            formatted_part = f"""[Source {i+1}: {source} (Relevance: {score:.2f})]
{text}
---"""
            formatted_parts.append(formatted_part)

        return "\n\n".join(formatted_parts)

    def _extract_sources(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source information from context items.

        Args:
            context: List of context items

        Returns:
            List of source dictionaries
        """
        sources = []
        seen_sources = set()

        for item in context:
            source = item.get("source", "Unknown")
            score = item.get("score", 0.0)
            final_score = item.get("final_score", score)

            if source not in seen_sources:
                source_info = {
                    "source": source,
                    "relevance_score": round(score, 3),
                    "final_score": round(final_score, 3),
                    "metadata": item.get("metadata", {}),
                }

                # Add source type
                if source.endswith(".pdf"):
                    source_info["type"] = "PDF Document"
                elif source.startswith("http"):
                    source_info["type"] = "Web Page"
                elif source.endswith((".docx", ".doc")):
                    source_info["type"] = "Word Document"
                else:
                    source_info["type"] = "Document"

                sources.append(source_info)
                seen_sources.add(source)

        # Sort by relevance score
        sources.sort(key=lambda x: x["final_score"], reverse=True)
        return sources

    def _calculate_confidence(self, context: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on context quality.

        Args:
            context: List of context items

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not context:
            return 0.0

        # Calculate average similarity score
        scores = [item.get("final_score", item.get("score", 0.0)) for item in context]
        avg_score = sum(scores) / len(scores)

        # Factor in the number of context items
        context_factor = min(len(context) / 3.0, 1.0)  # Normalize to max of 3 items

        # Factor in score distribution (prefer consistent scores)
        if len(scores) > 1:
            score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            consistency_factor = max(0.5, 1.0 - score_variance)
        else:
            consistency_factor = 1.0

        # Combine factors
        confidence = (
            (avg_score * 0.6) + (context_factor * 0.2) + (consistency_factor * 0.2)
        )

        return min(confidence, 1.0)

    def _assess_response_quality(
        self, response: str, query: str, context: List[Dict[str, Any]]
    ) -> float:
        """
        Assess the quality of the generated response.

        Args:
            response: Generated response
            query: Original query
            context: Context used for generation

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not response or len(response.strip()) < 10:
            return 0.1

        quality_score = 0.5  # Base score

        # Check response length (not too short, not too long)
        response_length = len(response)
        if 50 <= response_length <= 1000:
            quality_score += 0.2
        elif response_length > 1000:
            quality_score += 0.1

        # Check if response addresses the query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        word_overlap = len(query_words.intersection(response_words))
        if word_overlap > 0:
            quality_score += min(word_overlap / len(query_words), 0.2)

        # Check if response uses context information
        context_texts = [item.get("text", "") for item in context]
        context_words = set()
        for text in context_texts:
            context_words.update(text.lower().split())

        context_usage = len(response_words.intersection(context_words))
        if context_usage > 5:  # Uses substantial context
            quality_score += 0.1

        return min(quality_score, 1.0)

    def get_supported_models(self) -> List[str]:
        """
        Get list of supported models.

        Returns:
            List of available model names
        """
        models = ["fallback"]

        if self.gemini_client:
            models.extend(["gemini-2.5-flash-preview-05-20", "gemini-1.5-pro"])

        if self.openai_client:
            models.extend(["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])

        return models

    def update_model(self, model_name: str) -> bool:
        """
        Update the model used for generation.

        Args:
            model_name: Name of the model to use

        Returns:
            True if model was updated successfully
        """
        try:
            if model_name in self.get_supported_models():
                self.model = model_name
                self.logger.info(f"Model updated to: {model_name}")
                return True
            else:
                self.logger.warning(f"Model {model_name} not supported")
                return False
        except Exception as e:
            self.logger.error(f"❌ Error updating model: {e}")
            return False

    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about response generation.

        Returns:
            Dictionary with generation statistics
        """
        return {
            "supported_models": self.get_supported_models(),
            "current_model": self.model,
            "gemini_available": self.gemini_client is not None,
            "openai_available": self.openai_client is not None,
            "langchain_available": self.llm is not None,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
