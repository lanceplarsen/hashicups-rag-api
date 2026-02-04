from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging

from app.config import settings
from app.tracer import tracer, instrument_app
from app.metrics import metrics_endpoint, MetricsMiddleware
from app.handlers import health, chat
from app.vectorstore.chroma_store import ChromaStore
from app.client.anthropic_client import AnthropicClient
from app.services.indexer import IndexerService
from app.services.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
chroma_store: ChromaStore = None
anthropic_client: AnthropicClient = None
indexer_service: IndexerService = None
rag_pipeline: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global chroma_store, anthropic_client, indexer_service, rag_pipeline

    logger.info("Starting HashiCups RAG API")

    # Initialize Chroma store
    logger.info("Initializing Chroma vector store...")
    chroma_store = ChromaStore()
    chroma_store.initialize()

    # Initialize Anthropic client
    logger.info("Initializing Anthropic client...")
    anthropic_client = AnthropicClient()

    # Initialize RAG pipeline
    logger.info("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline(chroma_store, anthropic_client)

    # Initialize and start indexer
    logger.info("Starting indexer service...")
    indexer_service = IndexerService(chroma_store)
    await indexer_service.start()

    # Set dependencies for handlers
    health.set_dependencies(indexer_service, None, anthropic_client)
    chat.set_dependencies(rag_pipeline)

    logger.info("HashiCups RAG API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down HashiCups RAG API")
    if indexer_service:
        await indexer_service.stop()
    logger.info("HashiCups RAG API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="HashiCups RAG API",
    description="RAG-powered coffee recommendation and Q&A service",
    version="1.0.0",
    lifespan=lifespan
)

# Instrument the app for distributed tracing
instrument_app(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

# Include routers
app.include_router(health.router)
app.include_router(chat.router)


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return await metrics_endpoint()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "hashicups-rag-api",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat",
            "health": "/health/livez",
            "readiness": "/health/readyz",
            "stats": "/health/stats",
            "metrics": "/metrics"
        }
    }


def main():
    """Run the application."""
    host, port = settings.bind_address.split(":")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=int(port),
        log_level="info"
    )


if __name__ == "__main__":
    main()
