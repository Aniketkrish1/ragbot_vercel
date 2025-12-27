"""
Configuration file for RAG Chatbot
"""
import os
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration settings for RAG system"""
    
    # Paths
    VECTORDB_PATH: str = "vectorstore"
    
    # Embedding model (HuggingFace Inference API - free tier)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    USE_HF_INFERENCE_API: bool = True  # Using HF API to avoid local storage
    
    # Text chunking
    CHUNK_SIZE: int = 500  # Reduced to prevent token limit issues
    CHUNK_OVERLAP: int = 100
    
    # Retrieval
    TOP_K_DOCUMENTS: int = 2  # Reduced to 2 to fit within 6000 token limit
    
    # LLM settings
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.3 # Slightly higher for more natural flow (0.1 is too robotic)
    LLM_MAX_TOKENS: int = 1024
    
    # System prompt for smart context-aware responses
    SYSTEM_PROMPT: str = """You are a helpful AI Financial Assistant for FinProHub.
Your goal is to help users understand the provided financial documents in a natural, conversational way.

Guidelines:
1. **Be Natural**: Talk like a helpful human expert.
2. **Be Concise**: Keep answers clear and to the point.
3. **Use Context**: Base your answers on the provided context. If the answer isn't there, say you don't know politely.
4. **Friendly Tone**: Be professional but approachable."""

config = Config()
