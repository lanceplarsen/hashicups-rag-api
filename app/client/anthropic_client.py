import anthropic
import asyncio
import contextvars
import time
from typing import List, Optional

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

Use the coffee catalog below to make personalized recommendations, but don't just list
products - be a real barista who connects with customers. Keep responses concise and warm.

If someone asks something completely off-topic (politics, tech support, etc.), gently
steer back: "Ha, I'm more of a coffee expert than a {topic} expert! Speaking of
pick-me-ups though..." """

INTENT_CLASSIFICATION_PROMPT = """Classify this coffee shop customer message into exactly one category:

- coffee_search: Looking for specific coffee by flavor, origin, ingredients, or type
- conversational: Mood, greeting, occasion, or chat that could relate to coffee recommendations
- off_topic: Completely unrelated to coffee or a coffee shop visit

Message: "{message}"

Reply with only the category name, nothing else."""

QUERY_REWRITE_PROMPT = """Rewrite the customer's latest message as a standalone coffee search query.
Include relevant context from the conversation so the query makes sense on its own.

Conversation:
{history}

Latest message: "{message}"

Rewritten search query (just the query, nothing else):"""


class AnthropicClient:
    """Claude API client with tracing and metrics."""

    INTENT_MODEL = "claude-3-haiku-20240307"

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model
        self.max_tokens = settings.anthropic_max_tokens

    async def classify_intent(self, message: str) -> str:
        """Classify user message intent using Haiku for fast/cheap classification."""
        with tracer.start_as_current_span("anthropic.classify_intent") as span:
            span.set_attribute("llm.model", self.INTENT_MODEL)
            span.set_attribute("message", message)

            try:
                ctx = contextvars.copy_context()
                def _do_classify():
                    return self.client.messages.create(
                        model=self.INTENT_MODEL,
                        max_tokens=20,
                        messages=[{
                            "role": "user",
                            "content": INTENT_CLASSIFICATION_PROMPT.format(message=message)
                        }]
                    )
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ctx.run(_do_classify)
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

                ctx = contextvars.copy_context()
                def _do_rewrite():
                    return self.client.messages.create(
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
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ctx.run(_do_rewrite)
                )

                rewritten = response.content[0].text.strip()
                span.set_attribute("rewritten_query", rewritten)
                return rewritten

            except Exception as e:
                span.record_exception(e)
                return message  # fallback to original on error

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
                # Build the full system prompt with context
                full_system = f"{SYSTEM_PROMPT}\n\n## Available Coffee Catalog:\n\n{context}"

                # Convert conversation history to Anthropic format
                anthropic_messages = []
                for msg in messages:
                    anthropic_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })

                # Add the current user message
                anthropic_messages.append({
                    "role": "user",
                    "content": current_message
                })

                # Call Claude API (synchronous client, run in thread pool).
                # Propagate trace context into the executor so the outbound HTTP span
                # is linked to this trace instead of starting a new one.
                ctx = contextvars.copy_context()
                def _do_create():
                    return self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=full_system,
                        messages=anthropic_messages
                    )
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ctx.run(_do_create)
                )

                duration = time.time() - start_time

                # Extract response text
                response_text = response.content[0].text

                # Extract token usage
                token_usage = TokenUsage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens
                )

                # Record metrics
                llm_requests_total.labels(
                    model=self.model,
                    status="success"
                ).inc()

                llm_tokens_total.labels(
                    model=self.model,
                    type="input"
                ).inc(token_usage.input_tokens)

                llm_tokens_total.labels(
                    model=self.model,
                    type="output"
                ).inc(token_usage.output_tokens)

                llm_request_duration_seconds.observe(duration)

                span.set_attribute("llm.input_tokens", token_usage.input_tokens)
                span.set_attribute("llm.output_tokens", token_usage.output_tokens)
                span.set_attribute("llm.duration_seconds", duration)

                anthropic_api_health.set(1)

                return response_text, token_usage

            except Exception as e:
                duration = time.time() - start_time
                llm_requests_total.labels(
                    model=self.model,
                    status="error"
                ).inc()
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
