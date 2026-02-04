from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositeHTTPPropagator
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from app.config import settings


def setup_tracing():
    """Configure OpenTelemetry tracing with OTLP exporter."""

    # Create resource with service information
    resource = Resource.create({
        "service.name": settings.otel_service_name,
        "service.version": settings.otel_service_version,
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Configure OTLP HTTP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=settings.otel_exporter_otlp_traces_endpoint,
    )

    # Add span processor
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    # Configure propagators (TraceContext + B3)
    set_global_textmap(
        CompositeHTTPPropagator([
            TraceContextTextMapPropagator(),
            B3MultiFormat(),
        ])
    )

    # Instrument HTTPX client for outgoing requests
    HTTPXClientInstrumentor().instrument()

    return trace.get_tracer(__name__)


def instrument_app(app):
    """Instrument a FastAPI app instance for tracing."""
    FastAPIInstrumentor.instrument_app(app)


tracer = setup_tracing()
