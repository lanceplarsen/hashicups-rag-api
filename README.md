# HashiCups RAG API

A RAG-powered coffee shop assistant for [HashiCups](https://github.com/hashicorp-demoapp). Customers ask natural language questions and get product recommendations backed by hybrid search (semantic + keyword) and LLM-generated tasting notes.

Built from scratch — no LangChain or LlamaIndex — for full control over retrieval scoring and observability.

## How It Works

```
User: "I want something velvety with chocolate notes"
                    |
                    v
         Intent Classifier (Haiku)
          /         |          \
   coffee_search  conversational  off_topic
         |
         v
   Query Rewriter (Haiku, if multi-turn)
         |
         v
   Hybrid Search ──────────────────────────────────
    |                                              |
    |  Semantic (60%)                   BM25 (40%) |
    |  SentenceTransformer              Field-weighted keyword
    |  cosine similarity                name:        3.0x
    |  via ChromaDB                     collection:  3.0x
    |                                   color:       2.5x
    |                                   ingredients: 2.0x
    |                                   content:     1.0x
    |______________________________________________|
                    |
                    v
           Score Fusion + Threshold (0.3)
                    |
                    v
          Claude (Sonnet) generates response
          with retrieved product context
                    |
                    v
   "The Connectaccino has a velvety texture with
    rich chocolate and caramel notes..."
```

### Intent Routing

The pipeline classifies every query before deciding how to handle it:

- **coffee_search** — specific product queries go through hybrid retrieval
- **conversational** — mood/vibe queries or menu-wide questions get the full catalog as context
- **off_topic** — non-coffee questions get a gentle redirect with product names

### LLM Enrichment at Index Time

Raw product data is sparse (names are HashiCorp puns, teasers are DevOps jokes). At index time, Haiku generates 2-3 sentence coffee-domain descriptions for each product covering roast level, flavor notes, mouthfeel, caffeine intensity, and occasion fit. These enrichments power both semantic embeddings and BM25 keyword matching, so queries about taste and mood work even though the original data doesn't contain that vocabulary.

Enrichments are cached with a fingerprint hash of product data — they only regenerate when a product changes.

### Hybrid Search

Semantic search handles paraphrasing and abstract queries ("quiet afternoon" finds cozy drinks). BM25 handles exact terms ("espresso" boosts products with espresso in their name and ingredients). The 60/40 split lets both contribute.

BM25 uses separate indexes per field with different weights. A product name match (3.0x) dominates over a description match (1.0x). No domain stop words — BM25's IDF naturally downweights terms that appear in every document while preserving discriminative value in fields where they don't. See [docs/hybrid-search-optimization.md](docs/hybrid-search-optimization.md) for the full writeup.

## API

### POST /chat

```json
{
  "message": "What are the tasting notes for the Connectaccino?",
  "conversation_history": [
    {"role": "user", "content": "I want an espresso"},
    {"role": "assistant", "content": "I'd recommend our Vagrante espresso..."}
  ]
}
```

```json
{
  "response": "The Connectaccino has lovely chocolate and caramel notes with a smooth, velvety mouthfeel...",
  "retrieved_coffees": [
    {
      "coffee": {
        "id": 7,
        "name": "Connectaccino",
        "teaser": "Discover the wonders of our meshy service",
        "price": 250,
        "ingredients": [
          {"id": 1, "name": "Espresso", "quantity": 40, "unit": "ml"},
          {"id": 2, "name": "Steamed Milk", "quantity": 300, "unit": "ml"}
        ],
        "enrichment": "The Connectaccino is a creamy and comforting interpretation of the classic cappuccino..."
      },
      "similarity_score": 0.5279
    }
  ],
  "token_usage": {"input_tokens": 856, "output_tokens": 124},
  "latency_ms": 1523.4
}
```

### Health & Monitoring

| Endpoint | Description |
|----------|-------------|
| `GET /health/livez` | Liveness probe |
| `GET /health/readyz` | Readiness probe (checks indexed documents) |
| `GET /health/stats` | Document count, index status, service health |
| `GET /metrics` | Prometheus metrics |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Claude API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Generation model |
| `PRODUCT_SERVICE_URI` | `http://product-api:9090` | Product Service for indexing |
| `PUBLIC_API_URI` | `http://public-api:8080/api` | Public API (GraphQL) |
| `CHROMA_PERSIST_DIR` | `/app/data/chroma` | Vector store persistence |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `RETRIEVAL_TOP_K` | `5` | Max documents to retrieve |
| `RETRIEVAL_THRESHOLD` | `0.3` | Min hybrid score to return |
| `INDEX_INTERVAL_SECONDS` | `300` | Background reindex frequency |
| `BIND_ADDRESS` | `0.0.0.0:8080` | App server address |
| `METRICS_ADDRESS` | `0.0.0.0:9102` | Prometheus metrics address |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `http://localhost:4318/v1/traces` | OTLP trace endpoint |

The API key can also be loaded from a file mount at `/etc/secrets/ANTHROPIC_API_KEY` (for Kubernetes).

## Development

### Local

```bash
make install
export ANTHROPIC_API_KEY=sk-ant-...
make run
```

### Docker

```bash
make docker-build
export ANTHROPIC_API_KEY=sk-ant-...
make docker-run
```

### With Docker Compose

```bash
cd ../hashicups-public-api/docker_compose
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up --build
```

### Quick Test

```bash
# Health
curl http://localhost:8080/health/stats

# Chat
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want something smooth and creamy"}'
```

## Observability

### Tracing

Every request is traced end-to-end with OpenTelemetry, viewable in Jaeger:

- Intent classification decision and latency
- Query rewriting (original vs rewritten)
- Semantic search candidates and distances
- Per-field BM25 scores, weights, and matched terms for each candidate
- Per-document indexed content and tokenized fields
- LLM generation with token usage
- Enrichment generation (cached vs generated counts)

The Docker image bundles an OpenTelemetry Collector that exports traces to Jaeger.

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rag_pipeline_duration_seconds` | Histogram | End-to-end RAG latency |
| `rag_retrieval_count` | Histogram | Documents retrieved per query |
| `rag_retrieval_duration_seconds` | Histogram | Vector search latency |
| `llm_requests_total` | Counter | Claude API calls (by model, status) |
| `llm_tokens_total` | Counter | Token usage (by model, input/output) |
| `llm_request_duration_seconds` | Histogram | LLM call latency |
| `vectorstore_documents_total` | Gauge | Indexed document count |
| `indexer_runs_total` | Counter | Index operations (by status) |
| `http_requests_total` | Counter | HTTP requests (by method, endpoint, status) |
| `http_request_duration_seconds` | Histogram | HTTP request latency |

## Project Structure

```
app/
  main.py                    # FastAPI app, startup/shutdown lifecycle
  config.py                  # Pydantic settings with env + file-mount secrets
  models.py                  # Request/response models
  metrics.py                 # Prometheus metric definitions
  tracer.py                  # OpenTelemetry setup
  handlers/
    chat.py                  # POST /chat endpoint
    health.py                # Health, readiness, stats endpoints
  client/
    anthropic_client.py      # Claude API (Sonnet + Haiku)
    product_service.py       # Product Service REST client
    public_api.py            # Public API GraphQL client
  services/
    rag_pipeline.py          # Intent routing, retrieval, generation
    indexer.py               # Background indexing + enrichment caching
  vectorstore/
    chroma_store.py          # ChromaDB + field-weighted BM25 hybrid search
docs/
  hybrid-search-optimization.md
Dockerfile                   # Multi-stage (OTel Collector + Python 3.11)
docker-entrypoint.sh         # Starts collector + app
Makefile                     # Build, run, test, docker targets
```

## Stack

- **Search**: ChromaDB (semantic) + BM25Okapi (keyword), SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: Claude Sonnet (generation), Claude Haiku (intent classification, query rewriting, enrichment)
- **API**: FastAPI + uvicorn
- **Observability**: OpenTelemetry + Jaeger (tracing), Prometheus (metrics)
- **Runtime**: Python 3.11, Docker (multi-arch amd64/arm64)
