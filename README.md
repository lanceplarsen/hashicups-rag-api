# HashiCups RAG API

RAG (Retrieval-Augmented Generation) service for natural language queries about the HashiCups coffee catalog.

## Features

- Natural language coffee recommendations using Claude
- Vector similarity search with Chroma
- Automatic coffee catalog indexing from Public API
- OpenTelemetry distributed tracing
- Prometheus metrics

## Architecture

```
User Query → Embed Query → Vector Search → Build Context → Claude Generation → Response
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BIND_ADDRESS` | `0.0.0.0:8080` | HTTP server address |
| `METRICS_ADDRESS` | `0.0.0.0:9102` | Prometheus metrics address |
| `PUBLIC_API_URI` | `http://public-api:8080/api` | GraphQL endpoint |
| `ANTHROPIC_API_KEY` | (required) | Claude API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Claude model |
| `CHROMA_PERSIST_DIR` | `/app/data/chroma` | Vector store path |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `RETRIEVAL_TOP_K` | `5` | Documents to retrieve |
| `INDEX_INTERVAL_SECONDS` | `300` | Re-index interval |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Natural language query |
| `/health/livez` | GET | Liveness probe |
| `/health/readyz` | GET | Readiness probe |
| `/health/stats` | GET | Service statistics |
| `/metrics` | GET | Prometheus metrics |

### Chat Request

```json
{
  "message": "What coffee is good for mornings?",
  "conversation_history": []
}
```

### Chat Response

```json
{
  "response": "For a morning pick-me-up, I'd recommend...",
  "retrieved_coffees": [
    {"coffee": {...}, "similarity_score": 0.82}
  ],
  "token_usage": {"input_tokens": 856, "output_tokens": 124},
  "latency_ms": 1523.4
}
```

## Development

### Local Setup

```bash
# Create virtual environment and install dependencies
make install

# Run with hot reload
export ANTHROPIC_API_KEY=sk-ant-...
make run
```

### Docker

```bash
# Build image
make docker-build

# Run container
export ANTHROPIC_API_KEY=sk-ant-...
make docker-run
```

### With Docker Compose

```bash
cd ../hashicups-public-api/docker_compose
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up --build
```

## Testing

```bash
# Health check
curl http://localhost:8080/health/livez

# Service stats
curl http://localhost:8080/health/stats

# Chat query
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What coffee has chocolate flavor?"}'

# Metrics
curl http://localhost:9102/metrics | grep rag
```

## Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rag_pipeline_duration_seconds` | Histogram | Total RAG latency |
| `rag_retrieval_count` | Histogram | Documents retrieved |
| `llm_requests_total` | Counter | Claude API calls |
| `llm_tokens_total` | Counter | Token usage |
| `vectorstore_documents_total` | Gauge | Indexed documents |
| `indexer_runs_total` | Counter | Index operations |
