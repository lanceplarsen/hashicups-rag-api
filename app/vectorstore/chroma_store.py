import chromadb
import json
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Optional, Dict
import os

from app.models import Coffee, CoffeeDocument, Ingredient
from app.config import settings
from app.tracer import tracer
from app.metrics import (
    vectorstore_documents_total,
    rag_retrieval_duration_seconds
)
import time


class ChromaStore:
    """Chroma vector store wrapper with hybrid search (semantic + field-weighted BM25)."""

    # Field weights for BM25 scoring (tune these for your use case)
    BM25_FIELD_WEIGHTS = {
        "name": 3.0,        # Highest: user searching for specific product
        "ingredients": 2.0, # High: ingredient searches are common
        "color": 2.5,       # High: color is a specific attribute search
        "content": 1.0      # Base: general description matches
    }

    def __init__(self):
        self.persist_dir = settings.chroma_persist_dir
        self.embedding_model_name = settings.embedding_model
        self.collection_name = "coffees"
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._embedding_model: Optional[SentenceTransformer] = None

        # Separate BM25 indexes per field for weighted scoring
        self._bm25_name: Optional[BM25Okapi] = None
        self._bm25_ingredients: Optional[BM25Okapi] = None
        self._bm25_color: Optional[BM25Okapi] = None
        self._bm25_content: Optional[BM25Okapi] = None
        self._bm25_doc_ids: List[str] = []  # Map BM25 index position to doc ID

        # Store tokenized corpora for BM25 and tracing
        self._corpus_name: List[List[str]] = []
        self._corpus_ingredients: List[List[str]] = []
        self._corpus_color: List[List[str]] = []
        self._corpus_content: List[List[str]] = []

    def initialize(self):
        """Initialize the Chroma client and embedding model."""
        with tracer.start_as_current_span("chroma.initialize") as span:
            # Ensure persist directory exists
            os.makedirs(self.persist_dir, exist_ok=True)

            # Initialize Chroma client with persistence (new API)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
            )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            # Load embedding model
            span.set_attribute("embedding.model", self.embedding_model_name)
            self._embedding_model = SentenceTransformer(self.embedding_model_name)

            span.set_attribute("collection.name", self.collection_name)

    # Map hex colors to human-readable names for search
    COLOR_NAME_MAP = {
        # HashiCups specific colors
        "#444": "dark gray charcoal",
        "#1fa7ee": "sky blue bright blue",
        "#ffd814": "yellow gold bright yellow",
        "#00ca8e": "green mint green teal",
        "#894bd1": "purple violet lavender",
        "#0e67ed": "blue bright blue royal blue",
        "#f44d8a": "pink magenta hot pink",
        "#f24c53": "red coral red bright red",
        "#14c6cb": "cyan teal turquoise aqua",
        # Reds
        "#ff0000": "red", "#8b0000": "dark red", "#dc143c": "crimson red",
        "#b22222": "firebrick red", "#cd5c5c": "indian red",
        # Greens
        "#00ff00": "green", "#008000": "dark green", "#006400": "forest green",
        "#228b22": "forest green", "#32cd32": "lime green", "#90ee90": "light green",
        "#98fb98": "pale green", "#2e8b57": "sea green", "#3cb371": "medium green",
        # Blues
        "#0000ff": "blue", "#000080": "navy blue", "#00008b": "dark blue",
        "#4169e1": "royal blue", "#1e90ff": "dodger blue", "#87ceeb": "sky blue",
        "#add8e6": "light blue", "#4682b4": "steel blue",
        # Browns/Tans (common for coffee)
        "#8b4513": "brown", "#a0522d": "sienna brown", "#d2691e": "chocolate brown",
        "#cd853f": "peru brown", "#deb887": "burlywood tan", "#f4a460": "sandy brown",
        "#d2b48c": "tan", "#bc8f8f": "rosy brown", "#c4a484": "caramel brown",
        "#6f4e37": "coffee brown", "#3c280d": "dark roast brown",
        # Blacks/Grays
        "#000000": "black", "#444444": "dark gray",
        "#808080": "gray", "#696969": "dim gray", "#a9a9a9": "dark silver",
        "#c0c0c0": "silver", "#d3d3d3": "light gray",
        # Oranges/Yellows
        "#ffa500": "orange", "#ff8c00": "dark orange", "#ff7f50": "coral orange",
        "#ffff00": "yellow", "#ffd700": "gold", "#f0e68c": "khaki yellow",
        # Purples
        "#800080": "purple", "#4b0082": "indigo purple", "#8b008b": "dark magenta",
        "#9932cc": "dark orchid purple", "#663399": "rebecca purple",
        # Whites/Creams
        "#ffffff": "white", "#fffaf0": "floral white", "#faf0e6": "linen cream",
        "#fff5ee": "seashell cream", "#f5f5dc": "beige cream", "#ffe4c4": "bisque cream",
    }

    def _get_color_name(self, hex_color: str) -> str:
        """Convert hex color to human-readable name."""
        if not hex_color:
            return ""
        # Normalize hex color (lowercase, handle 3-digit shorthand)
        normalized = hex_color.lower().strip()
        if normalized in self.COLOR_NAME_MAP:
            return self.COLOR_NAME_MAP[normalized]
        # Try to find a close match by checking if it starts with common prefixes
        # For unknown colors, try to describe based on RGB values
        if normalized.startswith("#"):
            try:
                hex_val = normalized.lstrip("#")
                if len(hex_val) == 3:
                    hex_val = "".join([c*2 for c in hex_val])
                r = int(hex_val[0:2], 16)
                g = int(hex_val[2:4], 16)
                b = int(hex_val[4:6], 16)
                # Simple heuristic for color naming
                if r > 200 and g > 200 and b > 200:
                    return "light cream"
                elif r < 50 and g < 50 and b < 50:
                    return "black"
                elif r > g and r > b:
                    return "reddish brown" if g > 50 else "red"
                elif g > r and g > b:
                    return "green"
                elif b > r and b > g:
                    return "blue"
                elif r > 100 and g > 50 and b < 100:
                    return "brown"
                else:
                    return "dark"
            except (ValueError, IndexError):
                pass
        return hex_color  # Return original if can't parse

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing. Lowercase and split on non-alphanumeric."""
        return re.findall(r'\w+', text.lower())

    def _build_searchable_text(self, metadata: dict, content: str) -> str:
        """Build searchable text combining name, ingredients, and content."""
        parts = [
            metadata.get("name", ""),
            metadata.get("ingredients", ""),
            content
        ]
        return " ".join(parts)

    def _format_ingredient(self, ing: Ingredient) -> str:
        """Format an ingredient with quantity and unit if available."""
        if ing.quantity and ing.unit:
            return f"{ing.quantity} {ing.unit} {ing.name}"
        elif ing.quantity:
            return f"{ing.quantity} {ing.name}"
        return ing.name

    def _coffee_to_document(self, coffee: Coffee) -> CoffeeDocument:
        """Convert a Coffee to a document for semantic embedding."""
        # Build ingredient list with quantities for embedding content
        ingredients_str = ", ".join([self._format_ingredient(ing) for ing in coffee.ingredients]) if coffee.ingredients else ""

        # Store ingredients as JSON for proper reconstruction
        ingredients_json = json.dumps([
            {"id": ing.id, "name": ing.name, "quantity": ing.quantity, "unit": ing.unit}
            for ing in coffee.ingredients
        ]) if coffee.ingredients else "[]"

        # Build natural content for embedding - no labels, no N/A, no price
        parts = [f"{coffee.name}: {coffee.teaser}"]

        if coffee.description:
            parts.append(coffee.description)

        if ingredients_str:
            parts.append(f"Made with {ingredients_str}.")

        if coffee.origin:
            parts.append(f"Origin: {coffee.origin}.")

        if coffee.collection:
            parts.append(f"Part of the {coffee.collection} collection.")

        # Convert hex color to human-readable name for embedding
        color_name = self._get_color_name(coffee.color) if coffee.color else ""
        if color_name:
            parts.append(f"Color: {color_name}.")

        content = " ".join(parts)

        return CoffeeDocument(
            id=str(coffee.id),
            content=content,
            metadata={
                "coffee_id": coffee.id,
                "name": coffee.name,
                "teaser": coffee.teaser,
                "description": coffee.description,
                "price": coffee.price,
                "image": coffee.image,
                "origin": coffee.origin or "",
                "collection": coffee.collection or "",
                "color": coffee.color or "",  # Keep original hex for API response
                "color_name": color_name,  # Human-readable for search/display
                "ingredients": ingredients_str,  # Formatted string for BM25
                "ingredients_json": ingredients_json  # JSON for reconstruction
            }
        )

    def index_coffees(self, coffees: List[Coffee]) -> int:
        """Index a list of coffees into the vector store."""
        with tracer.start_as_current_span("chroma.index_coffees") as span:
            span.set_attribute("coffees.count", len(coffees))

            if not coffees:
                return 0

            # Convert coffees to documents
            documents = [self._coffee_to_document(c) for c in coffees]

            # Generate embeddings
            contents = [doc.content for doc in documents]
            embeddings = self._embedding_model.encode(contents).tolist()

            # Clear existing collection and add new documents
            # Delete existing documents first
            existing_ids = self._collection.get()["ids"]
            if existing_ids:
                self._collection.delete(ids=existing_ids)

            # Add new documents
            self._collection.add(
                ids=[doc.id for doc in documents],
                embeddings=embeddings,
                documents=contents,
                metadatas=[doc.metadata for doc in documents]
            )

            # PersistentClient auto-persists, no need to call persist()

            # Build separate BM25 indexes for field-weighted scoring
            with tracer.start_as_current_span("chroma.build_bm25_indexes") as bm25_span:
                self._bm25_doc_ids = [doc.id for doc in documents]

                # Build separate corpora for each field
                self._corpus_name = []
                self._corpus_ingredients = []
                self._corpus_color = []
                self._corpus_content = []

                for doc in documents:
                    self._corpus_name.append(self._tokenize(doc.metadata.get("name", "")))
                    self._corpus_ingredients.append(self._tokenize(doc.metadata.get("ingredients", "")))
                    self._corpus_color.append(self._tokenize(doc.metadata.get("color_name", "")))
                    self._corpus_content.append(self._tokenize(doc.content))

                # Create BM25 index for each field
                self._bm25_name = BM25Okapi(self._corpus_name) if any(self._corpus_name) else None
                self._bm25_ingredients = BM25Okapi(self._corpus_ingredients) if any(self._corpus_ingredients) else None
                self._bm25_color = BM25Okapi(self._corpus_color) if any(self._corpus_color) else None
                self._bm25_content = BM25Okapi(self._corpus_content) if any(self._corpus_content) else None

                bm25_span.set_attribute("bm25.documents_indexed", len(self._bm25_doc_ids))
                bm25_span.set_attribute("bm25.field_weights", json.dumps(self.BM25_FIELD_WEIGHTS))
                bm25_span.set_attribute("bm25.avg_name_tokens",
                    sum(len(doc) for doc in self._corpus_name) / len(self._corpus_name) if self._corpus_name else 0)
                bm25_span.set_attribute("bm25.avg_ingredient_tokens",
                    sum(len(doc) for doc in self._corpus_ingredients) / len(self._corpus_ingredients) if self._corpus_ingredients else 0)
                bm25_span.set_attribute("bm25.avg_content_tokens",
                    sum(len(doc) for doc in self._corpus_content) / len(self._corpus_content) if self._corpus_content else 0)

            # Update metrics
            vectorstore_documents_total.set(len(documents))

            span.set_attribute("documents.indexed", len(documents))
            return len(documents)

    def _bm25_score(self, query: str, doc_id: str) -> Tuple[float, Dict[str, any], float]:
        """Calculate field-weighted BM25 score for a document.

        Returns (normalized_score 0-1, field_details dict, raw_weighted_score).
        """
        if doc_id not in self._bm25_doc_ids:
            return 0.0, {}, 0.0

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0, {}, 0.0

        doc_index = self._bm25_doc_ids.index(doc_id)

        # Score each field separately
        field_scores = {}
        field_matches = {}
        weighted_raw_score = 0.0

        # Name field
        if self._bm25_name:
            name_scores = self._bm25_name.get_scores(query_tokens)
            name_raw = name_scores[doc_index]
            name_matches = [t for t in query_tokens if t in set(self._corpus_name[doc_index])]
            field_scores["name"] = {"raw": name_raw, "weight": self.BM25_FIELD_WEIGHTS["name"]}
            field_matches["name"] = name_matches
            weighted_raw_score += name_raw * self.BM25_FIELD_WEIGHTS["name"]

        # Ingredients field
        if self._bm25_ingredients:
            ing_scores = self._bm25_ingredients.get_scores(query_tokens)
            ing_raw = ing_scores[doc_index]
            ing_matches = [t for t in query_tokens if t in set(self._corpus_ingredients[doc_index])]
            field_scores["ingredients"] = {"raw": ing_raw, "weight": self.BM25_FIELD_WEIGHTS["ingredients"]}
            field_matches["ingredients"] = ing_matches
            weighted_raw_score += ing_raw * self.BM25_FIELD_WEIGHTS["ingredients"]

        # Color field
        if self._bm25_color:
            color_scores = self._bm25_color.get_scores(query_tokens)
            color_raw = color_scores[doc_index]
            color_matches = [t for t in query_tokens if t in set(self._corpus_color[doc_index])]
            field_scores["color"] = {"raw": color_raw, "weight": self.BM25_FIELD_WEIGHTS["color"]}
            field_matches["color"] = color_matches
            weighted_raw_score += color_raw * self.BM25_FIELD_WEIGHTS["color"]

        # Content field
        if self._bm25_content:
            content_scores = self._bm25_content.get_scores(query_tokens)
            content_raw = content_scores[doc_index]
            content_matches = [t for t in query_tokens if t in set(self._corpus_content[doc_index])]
            field_scores["content"] = {"raw": content_raw, "weight": self.BM25_FIELD_WEIGHTS["content"]}
            field_matches["content"] = content_matches
            weighted_raw_score += content_raw * self.BM25_FIELD_WEIGHTS["content"]

        # Calculate max possible weighted score across all docs for normalization
        max_weighted = 0.0
        for i in range(len(self._bm25_doc_ids)):
            doc_weighted = 0.0
            if self._bm25_name:
                doc_weighted += self._bm25_name.get_scores(query_tokens)[i] * self.BM25_FIELD_WEIGHTS["name"]
            if self._bm25_ingredients:
                doc_weighted += self._bm25_ingredients.get_scores(query_tokens)[i] * self.BM25_FIELD_WEIGHTS["ingredients"]
            if self._bm25_color:
                doc_weighted += self._bm25_color.get_scores(query_tokens)[i] * self.BM25_FIELD_WEIGHTS["color"]
            if self._bm25_content:
                doc_weighted += self._bm25_content.get_scores(query_tokens)[i] * self.BM25_FIELD_WEIGHTS["content"]
            max_weighted = max(max_weighted, doc_weighted)

        normalized_score = weighted_raw_score / max_weighted if max_weighted > 0 else 0.0

        # Build detailed field info for tracing (ensure native Python types for JSON serialization)
        field_details = {
            "field_scores": {
                k: {"raw": float(round(v["raw"], 4)), "weighted": float(round(v["raw"] * v["weight"], 4))}
                for k, v in field_scores.items()
            },
            "field_matches": field_matches,
            "all_matched_terms": list(set(
                field_matches.get("name", []) +
                field_matches.get("ingredients", []) +
                field_matches.get("color", []) +
                field_matches.get("content", [])
            ))
        }

        return float(normalized_score), field_details, float(weighted_raw_score)

    def search(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[Tuple[Coffee, float]]:
        """Hybrid search combining semantic and keyword matching."""
        with tracer.start_as_current_span("chroma.hybrid_search") as span:
            start_time = time.time()

            top_k = top_k or settings.retrieval_top_k
            threshold = threshold or settings.retrieval_threshold

            span.set_attribute("query", query)
            span.set_attribute("top_k", top_k)
            span.set_attribute("threshold", threshold)
            span.set_attribute("semantic_weight", semantic_weight)
            span.set_attribute("keyword_weight", keyword_weight)

            # Step 1: Semantic search
            with tracer.start_as_current_span("chroma.semantic_search") as semantic_span:
                query_embedding = self._embedding_model.encode([query]).tolist()

                # Get more results than top_k to allow reranking
                results = self._collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(top_k * 2, self.get_document_count() or top_k),
                    include=["documents", "metadatas", "distances"]
                )
                semantic_span.set_attribute("candidates_retrieved", len(results["ids"][0]) if results["ids"] else 0)

            # Step 2: Calculate hybrid scores with field-weighted BM25
            all_candidates = []

            with tracer.start_as_current_span("chroma.bm25_scoring") as bm25_span:
                bm25_span.set_attribute("bm25.query_tokens", " ".join(self._tokenize(query)))
                bm25_span.set_attribute("bm25.field_weights", json.dumps(self.BM25_FIELD_WEIGHTS))

                if results["ids"] and results["ids"][0]:
                    for i, doc_id in enumerate(results["ids"][0]):
                        metadata = results["metadatas"][0][i]

                        # Semantic score (convert distance to similarity)
                        distance = results["distances"][0][i]
                        semantic_score = 1 - distance

                        # Field-weighted BM25 score
                        bm25_normalized, field_details, bm25_raw = self._bm25_score(query, doc_id)

                        # Combined hybrid score
                        hybrid_score = (semantic_score * semantic_weight) + (bm25_normalized * keyword_weight)

                        all_candidates.append({
                            "doc_id": doc_id,
                            "metadata": metadata,
                            "name": metadata["name"],
                            "semantic_score": float(round(semantic_score, 4)),
                            "bm25_score": float(round(bm25_normalized, 4)),
                            "bm25_raw": float(round(bm25_raw, 4)),
                            "field_scores": field_details.get("field_scores", {}),
                            "field_matches": field_details.get("field_matches", {}),
                            "matched_terms": field_details.get("all_matched_terms", []),
                            "hybrid_score": float(round(hybrid_score, 4)),
                            "passed_threshold": bool(hybrid_score >= threshold)
                        })

            # Step 3: Sort by hybrid score and apply threshold
            all_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

            # Detailed tracing
            span.set_attribute("candidates.count", len(all_candidates))
            if all_candidates:
                span.set_attribute("candidates.details", json.dumps([
                    {k: v for k, v in c.items() if k != "metadata"}
                    for c in all_candidates
                ]))

                # Summary stats
                span.set_attribute("score.semantic_best", max(c["semantic_score"] for c in all_candidates))
                span.set_attribute("score.bm25_best", max(c["bm25_score"] for c in all_candidates))
                span.set_attribute("score.bm25_raw_best", max(c["bm25_raw"] for c in all_candidates))
                span.set_attribute("score.hybrid_best", max(c["hybrid_score"] for c in all_candidates))

                # Field-level match summary
                name_matches = set()
                ingredient_matches = set()
                color_matches = set()
                content_matches = set()
                for c in all_candidates:
                    name_matches.update(c["field_matches"].get("name", []))
                    ingredient_matches.update(c["field_matches"].get("ingredients", []))
                    color_matches.update(c["field_matches"].get("color", []))
                    content_matches.update(c["field_matches"].get("content", []))

                span.set_attribute("bm25.name_matches", ", ".join(name_matches) if name_matches else "none")
                span.set_attribute("bm25.ingredient_matches", ", ".join(ingredient_matches) if ingredient_matches else "none")
                span.set_attribute("bm25.color_matches", ", ".join(color_matches) if color_matches else "none")
                span.set_attribute("bm25.content_matches", ", ".join(content_matches) if content_matches else "none")

            # Step 4: Build final results
            coffees_with_scores = []

            for candidate in all_candidates[:top_k]:
                if candidate["hybrid_score"] >= threshold:
                    metadata = candidate["metadata"]

                    # Parse ingredients from JSON (preferred) or fall back to string parsing
                    ingredients = []
                    ingredients_json = metadata.get("ingredients_json")
                    if ingredients_json:
                        try:
                            ingredients_data = json.loads(ingredients_json)
                            ingredients = [
                                Ingredient(
                                    id=ing.get("id", idx),
                                    name=ing.get("name", ""),
                                    quantity=ing.get("quantity"),
                                    unit=ing.get("unit")
                                )
                                for idx, ing in enumerate(ingredients_data)
                                if ing.get("name")
                            ]
                        except json.JSONDecodeError:
                            pass

                    coffee = Coffee(
                        id=metadata["coffee_id"],
                        name=metadata["name"],
                        teaser=metadata["teaser"],
                        description=metadata["description"],
                        price=metadata["price"],
                        image=metadata["image"],
                        origin=metadata.get("origin") or None,
                        collection=metadata.get("collection") or None,
                        color=metadata.get("color") or None,
                        ingredients=ingredients,
                    )
                    coffees_with_scores.append((coffee, candidate["hybrid_score"]))

            duration = time.time() - start_time
            rag_retrieval_duration_seconds.observe(duration)

            # Final tracing
            span.set_attribute("results.count", len(coffees_with_scores))
            span.set_attribute("search.duration_ms", round(duration * 1000, 2))

            matched_names = [c["name"] for c in all_candidates[:top_k] if c["passed_threshold"]]
            span.set_attribute("matches.names", ", ".join(matched_names) if matched_names else "none")

            return coffees_with_scores

    def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        if self._collection:
            return self._collection.count()
        return 0

    def get_all_coffees(self) -> List[Coffee]:
        """Retrieve all coffees from the store (for conversational context)."""
        with tracer.start_as_current_span("chroma.get_all_coffees") as span:
            if not self._collection:
                return []

            results = self._collection.get(include=["metadatas"])

            coffees = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]

                    # Parse ingredients from JSON (preferred) or fall back to empty
                    ingredients = []
                    ingredients_json = metadata.get("ingredients_json")
                    if ingredients_json:
                        try:
                            ingredients_data = json.loads(ingredients_json)
                            ingredients = [
                                Ingredient(
                                    id=ing.get("id", idx),
                                    name=ing.get("name", ""),
                                    quantity=ing.get("quantity"),
                                    unit=ing.get("unit")
                                )
                                for idx, ing in enumerate(ingredients_data)
                                if ing.get("name")
                            ]
                        except json.JSONDecodeError:
                            pass

                    coffee = Coffee(
                        id=metadata["coffee_id"],
                        name=metadata["name"],
                        teaser=metadata["teaser"],
                        description=metadata["description"],
                        price=metadata["price"],
                        image=metadata["image"],
                        origin=metadata.get("origin") or None,
                        collection=metadata.get("collection") or None,
                        color=metadata.get("color") or None,
                        ingredients=ingredients,
                    )
                    coffees.append(coffee)

            span.set_attribute("coffees.count", len(coffees))
            return coffees
