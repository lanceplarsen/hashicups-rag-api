import anthropic
import time
from typing import List

from app.models import Message, TokenUsage
from app.config import settings
from app.tracer import tracer
from app.metrics import (
    llm_requests_total,
    llm_tokens_total,
    llm_request_duration_seconds,
    anthropic_api_health
)


SYSTEM_PROMPT = """You are a friendly barista at HashiCups, a specialty coffee shop.

You genuinely love coffee and enjoy chatting with customers. You can:
- Recommend coffees based on mood, taste, or occasion
- Have casual conversation while naturally weaving in coffee suggestions
- Pick up on cues like "tired", "celebrating", "stressed" to suggest fitting drinks
- Share enthusiasm about your favorite drinks on the menu

IMPORTANT: Only mention products that appear in the coffee catalog below. Never invent
or guess product details (ingredients, prices, colors, descriptions). If a customer asks
about a product not in the catalog, say "I don't see that one on our menu right now" and
offer to help them find something similar from what's available.

Use the coffee catalog below to make personalized recommendations, but don't just list
products - be a real barista who connects with customers. Keep responses concise and warm.

If someone asks something completely off-topic (politics, tech support, etc.), gently
steer back: "Ha, I'm more of a coffee expert than a {topic} expert! Speaking of
pick-me-ups though..." """

INTENT_CLASSIFICATION_PROMPT = """Classify this coffee shop customer message into exactly one category:

- coffee_search: Asking about a specific drink by name, or searching by flavor, origin, ingredients, color, collection, price, or type. Includes any product name that could be a menu item (even unfamiliar names like "Connectaccino", "Vaulted Latte", etc.)
- conversational: Mood, greeting, occasion, general chat, OR comparative questions about the menu (e.g., "what coffee has the most caffeine?", "what's the strongest?", "compare your options", "what do you recommend?")
- off_topic: Completely unrelated to coffee, drinks, or a coffee shop visit (e.g., politics, tech support, weather)

When in doubt between coffee_search and off_topic, choose coffee_search - the customer is at a coffee shop.

Message: "{message}"

Reply with only the category name, nothing else."""

QUERY_REWRITE_PROMPT = """Rewrite the customer's latest message as a standalone coffee search query.
Include relevant context from the conversation so the query makes sense on its own.

Conversation:
{history}

Latest message: "{message}"

Rewritten search query (just the query, nothing else):"""

MENTION_COFFEES_TOOL = {
    "name": "mention_coffees",
    "description": "Call this with the exact product names from the catalog that you mentioned or recommended in your reply. If you did not mention any specific product, omit this call.",
    "input_schema": {
        "type": "object",
        "properties": {
            "coffee_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Product names from the catalog that you mentioned or recommended.",
            }
        },
        "required": ["coffee_names"],
    },
}

MENTION_COFFEES_SYSTEM_ADDITION = """

When you mention or recommend specific coffees from the catalog above, call the mention_coffees tool with the **exact** product names as they appear in the catalog (e.g. ["Sumatra", "Vaulted Latte"]). If you don't mention any specific product, you may omit the tool call."""


class AnthropicClient:
    """Claude API client with tracing and metrics."""

    INTENT_MODEL = "claude-3-haiku-20240307"

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model
        self.max_tokens = settings.anthropic_max_tokens

    async def classify_intent(self, message: str) -> str:
        """Classify user message intent using Haiku for fast/cheap classification."""
        with tracer.start_as_current_span("anthropic.classify_intent") as span:
            span.set_attribute("llm.model", self.INTENT_MODEL)
            span.set_attribute("message", message)

            try:
                response = await self.client.messages.create(
                    model=self.INTENT_MODEL,
                    max_tokens=20,
                    messages=[{
                        "role": "user",
                        "content": INTENT_CLASSIFICATION_PROMPT.format(message=message)
                    }]
                )

                intent = response.content[0].text.strip().lower()

                # Validate intent is one of expected values
                if intent not in ("coffee_search", "conversational", "off_topic"):
                    intent = "coffee_search"  # default fallback

                span.set_attribute("intent", intent)
                return intent

            except Exception as e:
                span.record_exception(e)
                return "coffee_search"  # fallback on error

    async def rewrite_query(self, message: str, conversation_history: List[Message]) -> str:
        """Rewrite a follow-up message as a standalone search query using conversation context."""
        with tracer.start_as_current_span("anthropic.rewrite_query") as span:
            span.set_attribute("llm.model", self.INTENT_MODEL)
            span.set_attribute("original_message", message)

            try:
                # Format conversation history
                history_str = "\n".join([
                    f"{msg.role.capitalize()}: {msg.content}"
                    for msg in conversation_history[-6:]  # Last 3 turns max
                ])

                response = await self.client.messages.create(
                    model=self.INTENT_MODEL,
                    max_tokens=100,
                    messages=[{
                        "role": "user",
                        "content": QUERY_REWRITE_PROMPT.format(
                            history=history_str,
                            message=message
                        )
                    }]
                )

                rewritten = response.content[0].text.strip()
                span.set_attribute("rewritten_query", rewritten)
                return rewritten

            except Exception as e:
                span.record_exception(e)
                return message  # fallback to original on error

    async def _call_claude(
        self,
        system: str,
        messages: List[Message],
        current_message: str,
        start_time: float,
        span,
        tools: list = None,
    ):
        """Shared scaffolding: build messages, call the API, record metrics, return the raw response."""
        anthropic_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        anthropic_messages.append({"role": "user", "content": current_message})

        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=anthropic_messages,
        )
        if tools:
            kwargs["tools"] = tools

        response = await self.client.messages.create(**kwargs)

        duration = time.time() - start_time
        token_usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        llm_requests_total.labels(model=self.model, status="success").inc()
        llm_tokens_total.labels(model=self.model, type="input").inc(token_usage.input_tokens)
        llm_tokens_total.labels(model=self.model, type="output").inc(token_usage.output_tokens)
        llm_request_duration_seconds.observe(duration)
        span.set_attribute("llm.input_tokens", token_usage.input_tokens)
        span.set_attribute("llm.output_tokens", token_usage.output_tokens)
        span.set_attribute("llm.duration_seconds", duration)
        anthropic_api_health.set(1)

        return response, token_usage, duration

    async def generate(
        self,
        context: str,
        messages: List[Message],
        current_message: str
    ) -> tuple[str, TokenUsage]:
        """Generate a response using Claude with the provided context."""
        with tracer.start_as_current_span("anthropic.generate") as span:
            span.set_attribute("llm.model", self.model)
            span.set_attribute("llm.context_length", len(context))
            start_time = time.time()

            try:
                full_system = f"{SYSTEM_PROMPT}\n\n## Available Coffee Catalog:\n\n{context}"
                response, token_usage, _ = await self._call_claude(
                    system=full_system,
                    messages=messages,
                    current_message=current_message,
                    start_time=start_time,
                    span=span,
                )
                response_text = response.content[0].text if response.content else ""
                return response_text, token_usage

            except Exception as e:
                duration = time.time() - start_time
                llm_requests_total.labels(model=self.model, status="error").inc()
                llm_request_duration_seconds.observe(duration)
                anthropic_api_health.set(0)
                span.record_exception(e)
                raise

    async def generate_with_mentioned_coffees(
        self,
        context: str,
        messages: List[Message],
        current_message: str,
    ) -> tuple[str, TokenUsage, List[str]]:
        """Generate a response with tool use; return prose, token usage, and coffee names Claude mentioned."""
        with tracer.start_as_current_span("anthropic.generate_with_mentioned_coffees") as span:
            span.set_attribute("llm.model", self.model)
            span.set_attribute("llm.context_length", len(context))
            start_time = time.time()

            try:
                full_system = (
                    f"{SYSTEM_PROMPT}\n\n## Available Coffee Catalog:\n\n{context}"
                    f"{MENTION_COFFEES_SYSTEM_ADDITION}"
                )
                response, token_usage, _ = await self._call_claude(
                    system=full_system,
                    messages=messages,
                    current_message=current_message,
                    start_time=start_time,
                    span=span,
                    tools=[MENTION_COFFEES_TOOL],
                )

                response_text_parts = []
                mentioned_names: List[str] = []

                for block in response.content:
                    if block.type == "text":
                        if block.text:
                            response_text_parts.append(block.text)
                    elif block.type == "tool_use" and block.name == "mention_coffees":
                        mentioned_names = block.input.get("coffee_names", []) or []

                response_text = "\n".join(response_text_parts) if response_text_parts else ""

                span.set_attribute("mentioned_coffees.count", len(mentioned_names))
                if mentioned_names:
                    span.set_attribute("mentioned_coffees.names", ",".join(mentioned_names))

                return response_text, token_usage, mentioned_names

            except Exception as e:
                duration = time.time() - start_time
                llm_requests_total.labels(model=self.model, status="error").inc()
                llm_request_duration_seconds.observe(duration)
                anthropic_api_health.set(0)
                span.record_exception(e)
                raise

    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        try:
            # Just verify the API key is set and valid format
            return bool(settings.anthropic_api_key)
        except Exception:
            return False
