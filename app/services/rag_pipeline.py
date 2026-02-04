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
        """Execute the full RAG pipeline."""
        with tracer.start_as_current_span("rag_pipeline.query") as span:
            start_time = time.time()

            span.set_attribute("query.message", message)
            span.set_attribute("query.history_length", len(conversation_history or []))

            conversation_history = conversation_history or []

            # Step 1: Retrieve relevant coffees
            with tracer.start_as_current_span("rag_pipeline.retrieve"):
                retrieved = self.chroma_store.search(message)
                rag_retrieval_count.observe(len(retrieved))

            span.set_attribute("retrieval.count", len(retrieved))

            # Early exit when no coffees match — skip LLM call to save latency and cost
            if not retrieved:
                latency_ms = (time.time() - start_time) * 1000
                rag_pipeline_duration_seconds.observe(latency_ms / 1000)
                span.set_attribute("response.length", 0)
                span.set_attribute("latency_ms", latency_ms)
                return ChatResponse(
                    response="We couldn't find any coffees that match that. Try describing taste, origin, or style (e.g. espresso, light roast, fruity).",
                    retrieved_coffees=[],
                    token_usage=TokenUsage(input_tokens=0, output_tokens=0),
                    latency_ms=round(latency_ms, 2)
                )

            # Step 2: Build context from retrieved documents
            context = self._build_context(retrieved)

            # Step 3: Generate response with Claude
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

            # Build response
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
        """Build context string from retrieved coffees."""
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
