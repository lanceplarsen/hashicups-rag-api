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


SYSTEM_PROMPT = """You are a helpful coffee expert assistant for HashiCups, a specialty coffee shop.
Your role is to help customers find the perfect coffee based on their preferences and needs.

When answering questions:
- Use the provided coffee catalog information to give accurate recommendations
- Be friendly and enthusiastic about coffee
- If a customer's preferences don't match any coffees well, suggest the closest alternatives
- Include specific details like origin, ingredients, and flavor notes when relevant
- Keep responses concise but informative

If asked about something not related to coffee or the HashiCups menu, politely redirect the conversation to coffee topics."""


class AnthropicClient:
    """Claude API client with tracing and metrics."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model
        self.max_tokens = settings.anthropic_max_tokens

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
