from typing import List
import time

from app.models import (
    Message,
    ChatResponse,
    RetrievedCoffee,
    TokenUsage
)
from app.client.anthropic_client import AnthropicClient
from app.vectorstore.chroma_store import ChromaStore
from app.tracer import tracer
from app.metrics import (
    rag_pipeline_duration_seconds,
    rag_retrieval_count
)


class RAGPipeline:
    """RAG pipeline orchestrating retrieval and generation."""

    def __init__(self, chroma_store: ChromaStore, anthropic_client: AnthropicClient):
        self.chroma_store = chroma_store
        self.anthropic_client = anthropic_client

    async def query(
        self,
        message: str,
        conversation_history: List[Message] = None
    ) -> ChatResponse:
        """Execute the full RAG pipeline with intent-based routing."""
        with tracer.start_as_current_span("rag_pipeline.query") as span:
            start_time = time.time()

            span.set_attribute("query.message", message)
            span.set_attribute("query.history_length", len(conversation_history or []))

            conversation_history = conversation_history or []

            # Step 1: Classify intent using Haiku (fast & cheap)
            with tracer.start_as_current_span("rag_pipeline.classify_intent"):
                intent = await self.anthropic_client.classify_intent(message)

            span.set_attribute("intent", intent)

            # Step 2: Route based on intent
            retrieved = []
            search_query = message

            if intent == "coffee_search":
                # Rewrite query if there's conversation history (for multi-turn context)
                if conversation_history:
                    with tracer.start_as_current_span("rag_pipeline.rewrite_query"):
                        search_query = await self.anthropic_client.rewrite_query(
                            message, conversation_history
                        )
                    span.set_attribute("rewritten_query", search_query)

                # Do semantic search for specific coffee queries
                with tracer.start_as_current_span("rag_pipeline.retrieve"):
                    retrieved = self.chroma_store.search(search_query)
                    rag_retrieval_count.observe(len(retrieved))
                span.set_attribute("retrieval.count", len(retrieved))

            # Step 3: Build context based on intent and retrieval results
            if intent == "coffee_search" and retrieved:
                # Use retrieved coffees for targeted recommendations
                context = self._build_context(retrieved)
            elif intent == "off_topic":
                # Minimal context for off-topic messages
                context = "Customer is asking about something unrelated to coffee."
            else:
                # For conversational messages or coffee_search with no results,
                # provide full catalog so Claude can make natural suggestions
                all_coffees = self.chroma_store.get_all_coffees()
                context = self._build_catalog_context(all_coffees)

            # Step 4: Generate response with Claude
            response_text, token_usage = await self.anthropic_client.generate(
                context=context,
                messages=conversation_history,
                current_message=message
            )

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            rag_pipeline_duration_seconds.observe(latency_ms / 1000)

            span.set_attribute("response.length", len(response_text))
            span.set_attribute("latency_ms", latency_ms)

            # Build response - include retrieved coffees for coffee_search,
            # or top suggestions for conversational
            retrieved_coffees = [
                RetrievedCoffee(coffee=coffee, similarity_score=round(score, 4))
                for coffee, score in retrieved
            ]

            return ChatResponse(
                response=response_text,
                retrieved_coffees=retrieved_coffees,
                token_usage=token_usage,
                latency_ms=round(latency_ms, 2)
            )

    def _build_context(self, retrieved: list) -> str:
        """Build context string from retrieved coffees with relevance scores."""
        if not retrieved:
            return "No matching coffees found in the catalog."

        context_parts = []
        for coffee, score in retrieved:
            ingredients_str = ", ".join([ing.name for ing in coffee.ingredients]) if coffee.ingredients else "N/A"

            coffee_info = f"""---
**{coffee.name}** (Relevance: {score:.0%})
- Teaser: {coffee.teaser}
- Description: {coffee.description}
- Origin: {coffee.origin or 'N/A'}
- Collection: {coffee.collection or 'N/A'}
- Ingredients: {ingredients_str}
- Price: ${coffee.price / 100:.2f}
"""
            context_parts.append(coffee_info)

        return "\n".join(context_parts)

    def _build_catalog_context(self, coffees: list) -> str:
        """Build context string from full coffee catalog (for conversational messages)."""
        if not coffees:
            return "The coffee catalog is currently empty."

        context_parts = ["Here's our full menu - feel free to suggest any that fit the conversation:\n"]
        for coffee in coffees:
            ingredients_str = ", ".join([ing.name for ing in coffee.ingredients]) if coffee.ingredients else "N/A"

            coffee_info = f"""---
**{coffee.name}**
- Teaser: {coffee.teaser}
- Ingredients: {ingredients_str}
- Price: ${coffee.price / 100:.2f}
"""
            context_parts.append(coffee_info)

        return "\n".join(context_parts)
