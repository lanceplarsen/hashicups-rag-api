from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time


# HTTP Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# RAG Pipeline metrics
rag_pipeline_duration_seconds = Histogram(
    'rag_pipeline_duration_seconds',
    'Total RAG pipeline latency',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

rag_retrieval_count = Histogram(
    'rag_retrieval_count',
    'Number of documents retrieved per query',
    buckets=[0, 1, 2, 3, 4, 5, 10]
)

rag_retrieval_duration_seconds = Histogram(
    'rag_retrieval_duration_seconds',
    'Vector retrieval latency',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# LLM metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total Claude API calls',
    ['model', 'status']
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['model', 'type']
)

llm_request_duration_seconds = Histogram(
    'llm_request_duration_seconds',
    'Claude API request latency',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Vector store metrics
vectorstore_documents_total = Gauge(
    'vectorstore_documents_total',
    'Total documents in vector store'
)

# Indexer metrics
indexer_runs_total = Counter(
    'indexer_runs_total',
    'Total indexing operations',
    ['status']
)

indexer_duration_seconds = Histogram(
    'indexer_duration_seconds',
    'Indexing operation latency',
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0]
)

# Public API metrics
public_api_requests_total = Counter(
    'public_api_requests_total',
    'Total requests to Public API',
    ['endpoint', 'status']
)

public_api_request_duration_seconds = Histogram(
    'public_api_request_duration_seconds',
    'Public API request latency',
    ['endpoint']
)

# Health metrics
service_health = Gauge(
    'service_health',
    'Service health status (1=healthy, 0=unhealthy)'
)

public_api_health = Gauge(
    'public_api_health',
    'Public API health status (1=healthy, 0=unhealthy)'
)

anthropic_api_health = Gauge(
    'anthropic_api_health',
    'Anthropic API health status (1=healthy, 0=unhealthy)'
)


async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


class MetricsMiddleware:
    """Middleware to track HTTP request metrics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        path = scope["path"]

        # Skip metrics endpoint
        if path == "/metrics":
            await self.app(scope, receive, send)
            return

        start_time = time.time()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status = message["status"]
                duration = time.time() - start_time

                http_requests_total.labels(
                    method=method,
                    endpoint=path,
                    status=status
                ).inc()

                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=path
                ).observe(duration)

            await send(message)

        await self.app(scope, receive, send_wrapper)
