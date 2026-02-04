import httpx
from typing import List, Optional
import asyncio
import time

from app.models import Coffee, Ingredient
from app.config import settings
from app.tracer import tracer
from app.metrics import (
    public_api_requests_total,
    public_api_request_duration_seconds,
    public_api_health
)


# GraphQL query to fetch all coffees
COFFEES_QUERY = """
query GetCoffees {
    coffees {
        id
        name
        teaser
        description
        price
        image
    }
}
"""

# GraphQL query to fetch ingredients for a specific coffee
INGREDIENTS_QUERY = """
query GetCoffeeIngredients($coffeeID: String!) {
    coffeeIngredients(coffeeID: $coffeeID) {
        name
        quantity
        unit
    }
}
"""


class PublicAPIClient:
    """GraphQL client for Public API with tracing."""

    def __init__(self):
        self.base_url = settings.public_api_uri
        self.timeout = settings.public_api_timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=self.timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _get_ingredients(self, coffee_id: str) -> List[Ingredient]:
        """Fetch ingredients for a specific coffee."""
        try:
            response = await self._client.post(
                self.base_url,
                json={
                    "query": INGREDIENTS_QUERY,
                    "variables": {"coffeeID": coffee_id}
                }
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                return []

            ingredients_data = data.get("data", {}).get("coffeeIngredients", []) or []

            return [
                Ingredient(
                    id=i,
                    name=ing.get("name") or "Unknown"
                )
                for i, ing in enumerate(ingredients_data)
                if ing.get("name")
            ]
        except Exception:
            return []

    async def get_coffees(self) -> List[Coffee]:
        """Fetch all coffees with ingredients from Public API via GraphQL."""
        with tracer.start_as_current_span("public_api.get_coffees") as span:
            start_time = time.time()

            try:
                # First, fetch all coffees
                response = await self._client.post(
                    self.base_url,
                    json={"query": COFFEES_QUERY}
                )
                duration = time.time() - start_time

                public_api_requests_total.labels(
                    endpoint="/api",
                    status=response.status_code
                ).inc()

                public_api_request_duration_seconds.labels(
                    endpoint="/api"
                ).observe(duration)

                response.raise_for_status()
                data = response.json()

                # Handle GraphQL errors
                if "errors" in data:
                    span.set_attribute("graphql.errors", str(data["errors"]))
                    public_api_health.set(0)
                    raise Exception(f"GraphQL errors: {data['errors']}")

                coffees_data = data.get("data", {}).get("coffees", [])

                # Fetch ingredients for all coffees concurrently
                async def build_coffee(c) -> Coffee:
                    coffee_id = str(c["id"])
                    ingredients = await self._get_ingredients(coffee_id)
                    return Coffee(
                        id=c["id"],
                        name=c["name"],
                        teaser=c.get("teaser") or "",
                        description=c.get("description") or "",
                        price=c.get("price") or 0.0,
                        image=c.get("image") or "",
                        origin=None,
                        collection=None,
                        ingredients=ingredients
                    )

                # Fetch ingredients concurrently for all coffees
                coffees = await asyncio.gather(*[
                    build_coffee(c) for c in coffees_data
                ])

                span.set_attribute("coffees.count", len(coffees))
                public_api_health.set(1)

                return list(coffees)

            except Exception as e:
                duration = time.time() - start_time
                public_api_requests_total.labels(
                    endpoint="/api",
                    status="error"
                ).inc()
                public_api_request_duration_seconds.labels(
                    endpoint="/api"
                ).observe(duration)
                public_api_health.set(0)
                span.record_exception(e)
                raise

    async def health_check(self) -> bool:
        """Check if Public API is healthy."""
        try:
            # Simple GraphQL introspection query
            response = await self._client.post(
                self.base_url,
                json={"query": "{ __typename }"}
            )
            return response.status_code == 200
        except Exception:
            return False
