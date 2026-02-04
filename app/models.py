from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Ingredient(BaseModel):
    """Coffee ingredient model."""
    id: int
    name: str


class Coffee(BaseModel):
    """Coffee product model."""
    id: int
    name: str
    teaser: str
    description: str
    price: float
    image: str
    origin: Optional[str] = None
    collection: Optional[str] = None
    ingredients: Optional[List[Ingredient]] = []


class Message(BaseModel):
    """Chat message model."""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    conversation_history: Optional[List[Message]] = []


class RetrievedCoffee(BaseModel):
    """Coffee with similarity score from retrieval."""
    coffee: Coffee
    similarity_score: float


class TokenUsage(BaseModel):
    """Token usage from LLM."""
    input_tokens: int
    output_tokens: int


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    retrieved_coffees: List[RetrievedCoffee]
    token_usage: TokenUsage
    latency_ms: float


class HealthStats(BaseModel):
    """Service statistics for health endpoint."""
    total_documents: int
    last_index_time: Optional[str] = None
    index_status: str
    public_api_healthy: bool
    anthropic_api_healthy: bool


class CoffeeDocument(BaseModel):
    """Document format for vector store."""
    id: str
    content: str
    metadata: Dict[str, Any]
