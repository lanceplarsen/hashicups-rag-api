import httpx
from typing import List, Optional
import asyncio
import time

from app.models import Coffee, Ingredient
from app.config import settings
from app.tracer import tracer
from app.metrics import (
    product_service_requests_total,
    product_service_request_duration_seconds,
    product_service_health
)


class ProductServiceClient:
    """REST client for Product Service (used for indexing)."""

    def __init__(self):
        self.base_url = settings.product_service_uri
        self.timeout = settings.product_service_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=self.timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _get_ingredients(self, coffee_id: int) -> List[Ingredient]:
        """Fetch ingredients for a specific coffee."""
        try:
            response = await self._client.get(
                f"{self.base_url}/coffees/{coffee_id}/ingredients"
            )
            response.raise_for_status()
            ingredients_data = response.json()

            return [
                Ingredient(
                    id=ing.get("id", i),
                    name=ing.get("name") or "Unknown",
                    quantity=ing.get("quantity"),
                    unit=ing.get("unit")
                )
                for i, ing in enumerate(ingredients_data)
                if ing.get("name")
            ]
        except Exception:
            return []

    async def get_coffees(self) -> List[Coffee]:
        """Fetch all coffees with ingredients from Product Service via REST."""
        with tracer.start_as_current_span("product_service.get_coffees") as span:
            start_time = time.time()

            try:
                # Fetch all coffees
                response = await self._client.get(f"{self.base_url}/coffees")
                duration = time.time() - start_time

                product_service_requests_total.labels(
                    endpoint="/coffees",
                    status=response.status_code
                ).inc()

                product_service_request_duration_seconds.labels(
                    endpoint="/coffees"
                ).observe(duration)

                response.raise_for_status()
                coffees_data = response.json()

                span.set_attribute("coffees.raw_count", len(coffees_data))

                # Fetch ingredients for all coffees concurrently
                async def build_coffee(c) -> Coffee:
                    coffee_id = c["id"]
                    ingredients = await self._get_ingredients(coffee_id)
                    return Coffee(
                        id=coffee_id,
                        name=c.get("name") or "",
                        teaser=c.get("teaser") or "",
                        description=c.get("description") or "",
                        price=c.get("price") or 0.0,
                        image=c.get("image") or "",
                        origin=c.get("origin"),
                        collection=c.get("collection"),
                        color=c.get("color"),
                        ingredients=ingredients
                    )

                # Fetch ingredients concurrently for all coffees
                coffees = await asyncio.gather(*[
                    build_coffee(c) for c in coffees_data
                ])

                span.set_attribute("coffees.count", len(coffees))
                product_service_health.set(1)

                return list(coffees)

            except Exception as e:
                duration = time.time() - start_time
                product_service_requests_total.labels(
                    endpoint="/coffees",
                    status="error"
                ).inc()
                product_service_request_duration_seconds.labels(
                    endpoint="/coffees"
                ).observe(duration)
                product_service_health.set(0)
                span.record_exception(e)
                raise

    async def health_check(self) -> bool:
        """Check if Product Service is healthy."""
        try:
            response = await self._client.get(f"{self.base_url}/coffees")
            return response.status_code == 200
        except Exception:
            return False
