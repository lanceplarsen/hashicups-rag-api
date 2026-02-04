from fastapi import APIRouter, HTTPException
from typing import TYPE_CHECKING

from app.models import ChatRequest, ChatResponse
from app.tracer import tracer

if TYPE_CHECKING:
    from app.services.rag_pipeline import RAGPipeline

router = APIRouter()

# User-facing message; never expose internal errors (API keys, request IDs, etc.)
CHAT_ERROR_MESSAGE = "Chat service is temporarily unavailable. Please try again later."

# This will be set by main.py
_rag_pipeline: "RAGPipeline" = None


def set_dependencies(rag_pipeline: "RAGPipeline"):
    """Set the dependencies for chat handler."""
    global _rag_pipeline
    _rag_pipeline = rag_pipeline


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for natural language queries about coffee.

    Takes a message and optional conversation history, retrieves relevant
    coffee information, and generates a response using Claude.
    """
    with tracer.start_as_current_span("chat.handler") as span:
        span.set_attribute("message.length", len(request.message))
        span.set_attribute("history.length", len(request.conversation_history or []))

        if not _rag_pipeline:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not initialized"
            )

        if not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )

        try:
            response = await _rag_pipeline.query(
                message=request.message,
                conversation_history=request.conversation_history or []
            )
            return response

        except Exception as e:
            span.record_exception(e)
            raise HTTPException(
                status_code=500,
                detail=CHAT_ERROR_MESSAGE
            )
