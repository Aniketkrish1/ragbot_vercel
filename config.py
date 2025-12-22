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
    
    # Embedding model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Text chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    
    # Retrieval
    TOP_K_DOCUMENTS: int = 3
    
    # LLM settings
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024
    
    # System prompt for smart context-aware responses
    SYSTEM_PROMPT: str = """You are a strategic advisor using documents as reference material, not scripts.

Use the provided document context and conversation history, but prioritize real-world effectiveness over literal accuracy.

When answering questions involving people or decisions:
1. Assess the human situation and emotional state.
2. Identify risks such as resistance, mistrust, or premature pitching.
3. Decide whether action should be taken now or delayed.
4. Only then suggest tactics, if appropriate.

Rules:
If a prospect signals being busy, distracted, hesitant, or unavailable:
- Do not reinforce benefits or restate the opportunity
- Do not send materials unless explicitly requested
- Do not attempt to preserve momentum through persuasion
- Reduce pressure and preserve optionality
- It is acceptable and sometimes preferable to pause or disengage
- Focus on building trust and rapport for future interactions

If information is missing from the documents, say so clearly.
Be concise, but do not remove necessary nuance.

"""

config = Config()
