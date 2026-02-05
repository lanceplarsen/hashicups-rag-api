import asyncio
import hashlib
import json
import os
from datetime import datetime
from typing import Optional, Dict
import logging

from app.client.product_service import ProductServiceClient
from app.client.anthropic_client import AnthropicClient
from app.models import Coffee
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

    def __init__(self, chroma_store: ChromaStore, anthropic_client: AnthropicClient = None):
        self.chroma_store = chroma_store
        self.anthropic_client = anthropic_client
        self.interval_seconds = settings.index_interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_index_time: Optional[datetime] = None
        self._index_status = "not_started"
        self._total_documents = 0
        self._enrichment_cache_path = os.path.join(settings.chroma_persist_dir, "enrichment_cache.json")

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

    def _coffee_fingerprint(self, coffee: Coffee) -> str:
        """Create a hash of coffee data to detect changes."""
        ingredients_str = ",".join(
            f"{ing.name}:{ing.quantity}:{ing.unit}" for ing in (coffee.ingredients or [])
        )
        data = f"{coffee.id}:{coffee.name}:{coffee.teaser}:{coffee.description}:{ingredients_str}"
        return hashlib.md5(data.encode()).hexdigest()

    def _load_enrichment_cache(self) -> Dict[str, str]:
        """Load cached enrichment descriptions from disk."""
        try:
            if os.path.exists(self._enrichment_cache_path):
                with open(self._enrichment_cache_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load enrichment cache: {e}")
        return {}

    def _save_enrichment_cache(self, cache: Dict[str, str]):
        """Save enrichment descriptions cache to disk."""
        try:
            os.makedirs(os.path.dirname(self._enrichment_cache_path), exist_ok=True)
            with open(self._enrichment_cache_path, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save enrichment cache: {e}")

    async def _enrich_coffees(self, coffees) -> Dict[int, str]:
        """Generate LLM coffee-domain descriptions, using cache for unchanged products."""
        if not self.anthropic_client:
            logger.info("No Anthropic client available, skipping enrichment")
            return {}

        with tracer.start_as_current_span("indexer.enrich_coffees") as span:
            cache = self._load_enrichment_cache()
            descriptions: Dict[int, str] = {}
            generated_count = 0
            cached_count = 0

            for coffee in coffees:
                fingerprint = self._coffee_fingerprint(coffee)
                cache_key = f"{coffee.id}:{fingerprint}"

                if cache_key in cache:
                    descriptions[coffee.id] = cache[cache_key]
                    cached_count += 1
                    continue

                # Generate new description
                ingredients_str = ", ".join(
                    f"{ing.quantity} {ing.unit} {ing.name}" if ing.quantity and ing.unit else ing.name
                    for ing in (coffee.ingredients or [])
                ) or "None listed"

                try:
                    desc = await self.anthropic_client.generate_coffee_description(
                        name=coffee.name,
                        teaser=coffee.teaser,
                        description=coffee.description or "",
                        ingredients=ingredients_str
                    )
                    if desc:
                        descriptions[coffee.id] = desc
                        cache[cache_key] = desc
                        generated_count += 1
                        logger.info(f"Generated enrichment for {coffee.name}: {desc[:80]}...")
                except Exception as e:
                    logger.warning(f"Failed to enrich {coffee.name}: {e}")

            self._save_enrichment_cache(cache)

            span.set_attribute("enrichment.generated", generated_count)
            span.set_attribute("enrichment.cached", cached_count)
            span.set_attribute("enrichment.total", len(descriptions))

            # Log each enriched description to the span for visibility in Jaeger
            for coffee in coffees:
                if coffee.id in descriptions:
                    span.set_attribute(
                        f"enrichment.{coffee.id}.{coffee.name}",
                        descriptions[coffee.id]
                    )

            logger.info(f"Enrichment complete: {generated_count} generated, {cached_count} cached")

            return descriptions

    async def _run_index(self):
        """Run a single indexing operation."""
        with tracer.start_as_current_span("indexer.run_index") as span:
            start_time = time.time()
            self._index_status = "indexing"

            try:
                logger.info("Starting indexing operation")

                # Fetch coffees from Product Service
                async with ProductServiceClient() as client:
                    coffees = await client.get_coffees()

                span.set_attribute("coffees.fetched", len(coffees))
                logger.info(f"Fetched {len(coffees)} coffees from Product Service")

                # Generate enriched descriptions for better semantic search
                enriched_descriptions = await self._enrich_coffees(coffees)
                span.set_attribute("enrichment.count", len(enriched_descriptions))

                # Index into Chroma
                indexed_count = self.chroma_store.index_coffees(coffees, enriched_descriptions)

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
