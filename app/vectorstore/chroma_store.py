import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
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
    """Chroma vector store wrapper with sentence-transformers embeddings."""

    def __init__(self):
        self.persist_dir = settings.chroma_persist_dir
        self.embedding_model_name = settings.embedding_model
        self.collection_name = "coffees"
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._embedding_model: Optional[SentenceTransformer] = None

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

    def _coffee_to_document(self, coffee: Coffee) -> CoffeeDocument:
        """Convert a Coffee to a document for indexing. Ingredients come first so they
        carry the most weight in the embedding for retrieval."""
        # Build ingredient list
        ingredients_str = ", ".join([ing.name for ing in coffee.ingredients]) if coffee.ingredients else "N/A"

        # Build document content: ingredients first (and repeated) for maximum retrieval weight
        content = f"""Ingredients: {ingredients_str}
Contains: {ingredients_str}

Name: {coffee.name}
Description: {coffee.description}
Teaser: {coffee.teaser}
Origin: {coffee.origin or 'N/A'}
Collection: {coffee.collection or 'N/A'}
Price: ${coffee.price / 100:.2f}"""

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
                "ingredients": ingredients_str
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

            # Update metrics
            vectorstore_documents_total.set(len(documents))

            span.set_attribute("documents.indexed", len(documents))
            return len(documents)

    def search(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None
    ) -> List[Tuple[Coffee, float]]:
        """Search for similar coffees based on query."""
        with tracer.start_as_current_span("chroma.search") as span:
            start_time = time.time()

            top_k = top_k or settings.retrieval_top_k
            threshold = threshold or settings.retrieval_threshold

            span.set_attribute("query", query)
            span.set_attribute("top_k", top_k)
            span.set_attribute("threshold", threshold)

            # Generate query embedding
            query_embedding = self._embedding_model.encode([query]).tolist()

            # Search
            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            duration = time.time() - start_time
            rag_retrieval_duration_seconds.observe(duration)

            # Convert results to Coffee objects with scores
            coffees_with_scores = []

            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    # Chroma returns distances, convert to similarity (1 - distance for cosine)
                    distance = results["distances"][0][i]
                    similarity = 1 - distance

                    # Apply threshold filter
                    if similarity >= threshold:
                        ingredients_str = metadata.get("ingredients") or ""
                        if ingredients_str and ingredients_str != "N/A":
                            ingredients = [
                                Ingredient(id=idx, name=part.strip())
                                for idx, part in enumerate(ingredients_str.split(","))
                                if part.strip()
                            ]
                        else:
                            ingredients = []

                        coffee = Coffee(
                            id=metadata["coffee_id"],
                            name=metadata["name"],
                            teaser=metadata["teaser"],
                            description=metadata["description"],
                            price=metadata["price"],
                            image=metadata["image"],
                            origin=metadata.get("origin") or None,
                            collection=metadata.get("collection") or None,
                            ingredients=ingredients,
                        )
                        coffees_with_scores.append((coffee, similarity))

            span.set_attribute("results.count", len(coffees_with_scores))
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
                    ingredients_str = metadata.get("ingredients") or ""
                    if ingredients_str and ingredients_str != "N/A":
                        ingredients = [
                            Ingredient(id=idx, name=part.strip())
                            for idx, part in enumerate(ingredients_str.split(","))
                            if part.strip()
                        ]
                    else:
                        ingredients = []

                    coffee = Coffee(
                        id=metadata["coffee_id"],
                        name=metadata["name"],
                        teaser=metadata["teaser"],
                        description=metadata["description"],
                        price=metadata["price"],
                        image=metadata["image"],
                        origin=metadata.get("origin") or None,
                        collection=metadata.get("collection") or None,
                        ingredients=ingredients,
                    )
                    coffees.append(coffee)

            span.set_attribute("coffees.count", len(coffees))
            return coffees

    def search_debug(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None
    ) -> dict:
        """Run a search and return raw Chroma results for debugging (no threshold filter).
        Returns document_count, query, top_k, threshold, and raw_results with id, distance,
        similarity, and name for each result Chroma returned."""
        top_k = top_k or settings.retrieval_top_k
        threshold = threshold or settings.retrieval_threshold
        document_count = self.get_document_count()

        query_embedding = self._embedding_model.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["metadatas", "distances"]
        )

        raw_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = 1 - distance
                raw_results.append({
                    "id": doc_id,
                    "distance": round(distance, 4),
                    "similarity": round(similarity, 4),
                    "name": metadata.get("name", ""),
                    "passed_threshold": similarity >= threshold,
                })

        return {
            "document_count": document_count,
            "query": query,
            "top_k": top_k,
            "threshold": threshold,
            "raw_results": raw_results,
            "passed_count": sum(1 for r in raw_results if r["passed_threshold"]),
        }
