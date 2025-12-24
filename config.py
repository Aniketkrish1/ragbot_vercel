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
    LLM_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 800  # Reduced response tokens to leave more room for context
    
    # System prompt for smart context-aware responses (optimized for token efficiency)
    SYSTEM_PROMPT: str = """You are a strategic advisor. Use provided context and prioritize effectiveness.

For questions involving decisions:
1. Assess the situation and risks
2. Decide if action is needed now or later
3. Suggest tactics only if appropriate

Key rule: If someone signals unavailability, reduce pressure and preserve trust. Don't force engagement.

Be concise. Say clearly if information is missing from documents."""

config = Config()
