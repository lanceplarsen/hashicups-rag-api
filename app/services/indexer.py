import asyncio
from datetime import datetime
from typing import Optional
import logging

from app.client.public_api import PublicAPIClient
from app.vectorstore.chroma_store import ChromaStore
from app.config import settings
from app.tracer import tracer
from app.metrics import (
    indexer_runs_total,
    indexer_duration_seconds,
    vectorstore_documents_total
)
import time

logger = logging.getLogger(__name__)


class IndexerService:
    """Background service for indexing coffee data."""

    def __init__(self, chroma_store: ChromaStore):
        self.chroma_store = chroma_store
        self.interval_seconds = settings.index_interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_index_time: Optional[datetime] = None
        self._index_status = "not_started"
        self._total_documents = 0

    @property
    def last_index_time(self) -> Optional[str]:
        """Get the last index time as ISO string."""
        if self._last_index_time:
            return self._last_index_time.isoformat()
        return None

    @property
    def index_status(self) -> str:
        """Get the current index status."""
        return self._index_status

    @property
    def total_documents(self) -> int:
        """Get the total number of indexed documents."""
        return self._total_documents

    async def start(self):
        """Start the background indexing service."""
        if self._running:
            return

        self._running = True
        logger.info("Starting indexer service")

        # Run initial index
        await self._run_index()

        # Start background task
        self._task = asyncio.create_task(self._background_loop())

    async def stop(self):
        """Stop the background indexing service."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Indexer service stopped")

    async def _background_loop(self):
        """Background loop for periodic indexing."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_seconds)
                if self._running:
                    await self._run_index()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in indexer background loop: {e}")
                self._index_status = f"error: {str(e)}"

    async def _run_index(self):
        """Run a single indexing operation."""
        with tracer.start_as_current_span("indexer.run_index") as span:
            start_time = time.time()
            self._index_status = "indexing"

            try:
                logger.info("Starting indexing operation")

                # Fetch coffees from Public API
                async with PublicAPIClient() as client:
                    coffees = await client.get_coffees()

                span.set_attribute("coffees.fetched", len(coffees))
                logger.info(f"Fetched {len(coffees)} coffees from Public API")

                # Index into Chroma
                indexed_count = self.chroma_store.index_coffees(coffees)

                duration = time.time() - start_time

                # Update state
                self._last_index_time = datetime.utcnow()
                self._index_status = "ready"
                self._total_documents = indexed_count

                # Record metrics
                indexer_runs_total.labels(status="success").inc()
                indexer_duration_seconds.observe(duration)
                vectorstore_documents_total.set(indexed_count)

                span.set_attribute("documents.indexed", indexed_count)
                span.set_attribute("duration_seconds", duration)

                logger.info(f"Indexing complete: {indexed_count} documents in {duration:.2f}s")

            except Exception as e:
                duration = time.time() - start_time
                self._index_status = f"error: {str(e)}"

                indexer_runs_total.labels(status="error").inc()
                indexer_duration_seconds.observe(duration)

                span.record_exception(e)
                logger.error(f"Indexing failed: {e}")
                raise

    async def force_reindex(self) -> int:
        """Force an immediate reindex operation."""
        await self._run_index()
        return self._total_documents
