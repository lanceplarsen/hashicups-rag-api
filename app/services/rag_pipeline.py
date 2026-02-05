from typing import List
import time

from app.models import (
    Message,
    ChatResponse,
    RetrievedCoffee,
    TokenUsage,
    Ingredient
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

    # Map hex colors to human-readable names for LLM context
    COLOR_NAME_MAP = {
        # HashiCups specific colors
        "#444": "dark gray",
        "#1fa7ee": "sky blue",
        "#ffd814": "bright yellow",
        "#00ca8e": "mint green",
        "#894bd1": "purple",
        "#0e67ed": "bright blue",
        "#f44d8a": "pink",
        "#f24c53": "red",
        "#14c6cb": "cyan",
        # Standard colors
        "#ff0000": "red", "#8b0000": "dark red", "#dc143c": "crimson",
        "#00ff00": "green", "#008000": "dark green", "#228b22": "forest green",
        "#0000ff": "blue", "#000080": "navy blue", "#87ceeb": "sky blue",
        "#8b4513": "brown", "#a0522d": "sienna brown", "#d2691e": "chocolate",
        "#6f4e37": "coffee brown", "#3c280d": "dark roast",
        "#000000": "black", "#444444": "dark gray",
        "#808080": "gray", "#c0c0c0": "silver",
        "#ffa500": "orange", "#ffd700": "gold",
        "#800080": "purple", "#4b0082": "indigo",
        "#ffffff": "white", "#f5f5dc": "beige", "#ffe4c4": "cream",
    }

    def __init__(self, chroma_store: ChromaStore, anthropic_client: AnthropicClient):
        self.chroma_store = chroma_store
        self.anthropic_client = anthropic_client

    def _get_color_name(self, hex_color: str) -> str:
        """Convert hex color to human-readable name."""
        if not hex_color:
            return "N/A"
        normalized = hex_color.lower().strip()
        if normalized in self.COLOR_NAME_MAP:
            return self.COLOR_NAME_MAP[normalized]
        # Fallback: try to parse RGB and describe
        if normalized.startswith("#"):
            try:
                hex_val = normalized.lstrip("#")
                if len(hex_val) == 3:
                    hex_val = "".join([c*2 for c in hex_val])
                r = int(hex_val[0:2], 16)
                g = int(hex_val[2:4], 16)
                b = int(hex_val[4:6], 16)
                if r > 200 and g > 200 and b > 200:
                    return "light cream"
                elif r < 50 and g < 50 and b < 50:
                    return "black"
                elif r > g and r > b:
                    return "reddish brown" if g > 50 else "red"
                elif g > r and g > b:
                    return "green"
                elif b > r and b > g:
                    return "blue"
                elif r > 100 and g > 50 and b < 100:
                    return "brown"
                return "dark"
            except (ValueError, IndexError):
                pass
        return hex_color

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
            all_coffees = []
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
                # Provide product names so Claude can redirect appropriately
                # but not full details (to discourage detailed responses about off-topic)
                all_coffees = self.chroma_store.get_all_coffees()
                product_names = ", ".join([c.name for c in all_coffees]) if all_coffees else "none available"
                context = f"""Customer is asking about something unrelated to coffee.

Gently redirect them back to coffee. Our menu includes: {product_names}

Do NOT invent details about products. Only reference products by name to redirect the conversation."""
            else:
                # For conversational messages or coffee_search with no results,
                # provide full catalog so Claude can make natural suggestions
                all_coffees = self.chroma_store.get_all_coffees()
                context = self._build_catalog_context(all_coffees)

            # Step 4: Generate response with Claude
            coffees: List[RetrievedCoffee] = []
            if intent == "coffee_search" and retrieved:
                response_text, token_usage, mentioned_names = await self.anthropic_client.generate_with_mentioned_coffees(
                    context=context,
                    messages=conversation_history,
                    current_message=message
                )
                coffees = self._filter_retrieved_by_mentioned(retrieved, mentioned_names)
            elif intent == "off_topic":
                response_text, token_usage = await self.anthropic_client.generate(
                    context=context,
                    messages=conversation_history,
                    current_message=message
                )
            else:
                response_text, token_usage, mentioned_names = await self.anthropic_client.generate_with_mentioned_coffees(
                    context=context,
                    messages=conversation_history,
                    current_message=message
                )
                coffees = self._resolve_mentioned_coffees(all_coffees, mentioned_names)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            rag_pipeline_duration_seconds.observe(latency_ms / 1000)

            span.set_attribute("response.length", len(response_text))
            span.set_attribute("latency_ms", latency_ms)

            return ChatResponse(
                response=response_text,
                retrieved_coffees=coffees,
                token_usage=token_usage,
                latency_ms=round(latency_ms, 2)
            )

    def _filter_retrieved_by_mentioned(
        self, retrieved: list, mentioned_names: List[str]
    ) -> List[RetrievedCoffee]:
        """Filter retrieved (coffee, score) to only mentioned names; preserve order and original scores."""
        if not retrieved or not mentioned_names:
            return []
        name_to_pair = {
            coffee.name.strip().lower(): (coffee, score)
            for coffee, score in retrieved
        }
        seen_ids = set()
        result = []
        for name in mentioned_names:
            key = (name or "").strip().lower()
            if not key:
                continue
            pair = name_to_pair.get(key)
            if pair:
                coffee, score = pair
                if coffee.id not in seen_ids:
                    seen_ids.add(coffee.id)
                    result.append(RetrievedCoffee(coffee=coffee, similarity_score=round(score, 4)))
        return result

    def _resolve_mentioned_coffees(
        self, catalog: list, mentioned_names: List[str]
    ) -> List[RetrievedCoffee]:
        """Map mentioned product names to catalog coffees; preserve order, dedupe by id."""
        if not catalog or not mentioned_names:
            return []
        name_to_coffee = {c.name.strip().lower(): c for c in catalog}
        seen_ids = set()
        result = []
        for name in mentioned_names:
            key = (name or "").strip().lower()
            if not key:
                continue
            coffee = name_to_coffee.get(key)
            if coffee and coffee.id not in seen_ids:
                seen_ids.add(coffee.id)
                result.append(RetrievedCoffee(coffee=coffee, similarity_score=1.0))
        return result

    def _format_ingredient(self, ing: Ingredient) -> str:
        """Format an ingredient with quantity and unit if available."""
        if ing.quantity and ing.unit:
            return f"{ing.quantity} {ing.unit} {ing.name}"
        elif ing.quantity:
            return f"{ing.quantity} {ing.name}"
        return ing.name

    def _build_context(self, retrieved: list) -> str:
        """Build context string from retrieved coffees with relevance scores."""
        if not retrieved:
            return "No matching coffees found in the catalog."

        context_parts = []
        for coffee, score in retrieved:
            ingredients_str = ", ".join([self._format_ingredient(ing) for ing in coffee.ingredients]) if coffee.ingredients else "N/A"

            coffee_info = f"""---
**{coffee.name}** (Relevance: {score:.0%})
- Teaser: {coffee.teaser}
- Description: {coffee.description}
- Origin: {coffee.origin or 'N/A'}
- Collection: {coffee.collection or 'N/A'}
- Color: {self._get_color_name(coffee.color)}
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
            ingredients_str = ", ".join([self._format_ingredient(ing) for ing in coffee.ingredients]) if coffee.ingredients else "N/A"

            coffee_info = f"""---
**{coffee.name}**
- Teaser: {coffee.teaser}
- Origin: {coffee.origin or 'N/A'}
- Collection: {coffee.collection or 'N/A'}
- Color: {self._get_color_name(coffee.color)}
- Ingredients: {ingredients_str}
- Price: ${coffee.price / 100:.2f}
"""
            context_parts.append(coffee_info)

        return "\n".join(context_parts)
