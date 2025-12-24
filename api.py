"""
FastAPI Backend for RAG Chatbot
Provides REST API endpoints for context-aware PDF question answering
"""

import os
import uuid
import time
import functools
import json
from typing import Optional, Dict, List, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Vercel KV support
try:
    import vercel_kv
    VERCEL_KV_AVAILABLE = True
except ImportError:
    VERCEL_KV_AVAILABLE = False

from config import config

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Context-aware PDF Question Answering with Groq LLM",
    version="1.0.0"
)

# CORS Configuration
# For development: allows all origins
# For production: replace "*" with your specific frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["https://yourdomain.com"] in production
    allow_credentials=True,  # Required for cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage (uses Vercel KV on Vercel, falls back to in-memory)
sessions: Dict[str, Dict] = {}

# Session configuration
SESSION_TIMEOUT = 3600  # 1 hour in seconds
CLEANUP_INTERVAL = 300  # Clean old sessions every 5 minutes
last_cleanup = time.time()

# Cache for loaded models (load once, reuse)
_vectorstore = None
_embeddings = None
_llm = None


# ============================================
# Request/Response Models
# ============================================

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="User question")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="LLM temperature")
    k_documents: Optional[int] = Field(default=None, ge=1, le=10, description="Number of context documents")


class SourceDocument(BaseModel):
    page: int
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    session_id: str
    memory_status: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    active_sessions: int


# ============================================
# Session Management
# ============================================

def get_session_data(session_id: str) -> Optional[Dict]:
    """Get session data from Vercel KV or in-memory storage"""
    if VERCEL_KV_AVAILABLE:
        try:
            data = vercel_kv.get(f"session:{session_id}")
            if data:
                return json.loads(data) if isinstance(data, str) else data
        except Exception as e:
            print(f"Error reading from Vercel KV: {e}")
    
    # Fallback to in-memory
    return sessions.get(session_id)


def set_session_data(session_id: str, data: Dict):
    """Save session data to Vercel KV or in-memory storage"""
    if VERCEL_KV_AVAILABLE:
        try:
            vercel_kv.set(f"session:{session_id}", json.dumps(data), ex=SESSION_TIMEOUT)
            return
        except Exception as e:
            print(f"Error writing to Vercel KV: {e}")
    
    # Fallback to in-memory
    sessions[session_id] = data


def delete_session_data(session_id: str):
    """Delete session data from Vercel KV or in-memory storage"""
    if VERCEL_KV_AVAILABLE:
        try:
            vercel_kv.delete(f"session:{session_id}")
            return
        except Exception as e:
            print(f"Error deleting from Vercel KV: {e}")
    
    # Fallback to in-memory
    if session_id in sessions:
        del sessions[session_id]


def cleanup_old_sessions():
    """Remove sessions that haven't been accessed in SESSION_TIMEOUT seconds"""
    global last_cleanup
    
    current_time = time.time()
    if current_time - last_cleanup < CLEANUP_INTERVAL:
        return
    
    last_cleanup = current_time
    
    # Only cleanup in-memory sessions (Vercel KV handles expiry automatically)
    expired_sessions = [
        sid for sid, data in sessions.items()
        if current_time - data.get("last_access", 0) > SESSION_TIMEOUT
    ]
    
    for sid in expired_sessions:
        del sessions[sid]
    
    if expired_sessions:
        print(f"Cleaned up {len(expired_sessions)} expired sessions")


def get_or_create_session(session_id: Optional[str]) -> str:
    """Get existing session or create new one"""
    cleanup_old_sessions()
    
    if session_id:
        session_data = get_session_data(session_id)
        if session_data:
            session_data["last_access"] = time.time()
            set_session_data(session_id, session_data)
            return session_id
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    set_session_data(new_session_id, {
        "history": [],
        "last_access": time.time(),
        "created_at": datetime.utcnow().isoformat()
    })
    return new_session_id


def get_session_history(session_id: str) -> List[str]:
    """Get chat history for session"""
    session_data = get_session_data(session_id)
    if session_data:
        return session_data.get("history", [])
    return []


def save_to_session(session_id: str, user_msg: str, ai_msg: str):
    """Save messages to session history"""
    session_data = get_session_data(session_id)
    if session_data:
        session_data["history"].extend([user_msg, ai_msg])
        session_data["last_access"] = time.time()
        set_session_data(session_id, session_data)


# ============================================
# Model Loading (with caching)
# ============================================

@functools.lru_cache(maxsize=1)
def load_embeddings():
    """Load embeddings model (cached)"""
    print("Loading embeddings model...")
    
    if config.USE_HF_INFERENCE_API:
        # Use HuggingFace Inference API (free tier, no local storage)
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN not found. Set USE_HF_INFERENCE_API=False in config.py to use local model.")
            raise RuntimeError("HF_TOKEN required when USE_HF_INFERENCE_API=True")
        
        embeddings = HuggingFaceEndpointEmbeddings(
            model=config.EMBEDDING_MODEL,
            huggingfacehub_api_token=hf_token
        )
        print("Embeddings loaded via HuggingFace API (no local storage)")
    else:
        # Use local model (requires ~80-90MB storage)
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embeddings loaded locally")
    
    return embeddings


def load_vectorstore():
    """Load FAISS vectorstore (cached)"""
    global _vectorstore, _embeddings
    
    if _vectorstore is not None:
        return _vectorstore
    
    if not os.path.exists(config.VECTORDB_PATH):
        raise RuntimeError(f"Vector store not found at {config.VECTORDB_PATH}. Run create_vectordb.py first.")
    
    print("Loading vector store...")
    try:
        _embeddings = load_embeddings()
        _vectorstore = FAISS.load_local(
            config.VECTORDB_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Vector store loaded successfully from {config.VECTORDB_PATH}")
        return _vectorstore
    except Exception as e:
        print(f"Error loading vector store: {e}")
        raise


def get_llm(temperature: float):
    """Get LLM instance (cached by temperature)"""
    return ChatGroq(
        model=config.LLM_MODEL,
        temperature=temperature,
        max_tokens=config.LLM_MAX_TOKENS,
        api_key=os.getenv("GROQ_API_KEY")
    )


# ============================================
# RAG Chain Creation
# ============================================

def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters for English)"""
    return len(text) // 4


def create_rag_chain(vectorstore, temperature: float, k_docs: int, session_history: List[str]):
    """Create conversational retrieval chain with memory"""
    
    # Trim history if it's too long (safety check for token limits)
    estimated_history_tokens = estimate_tokens(" ".join(session_history))
    if estimated_history_tokens > 1000:  # If history exceeds 1000 tokens
        # Keep only the most recent messages
        session_history = session_history[-6:]  # Last 3 exchanges (user + AI)
        print(f"Trimmed conversation history to prevent token limit issues")
    
    # Initialize LLM
    llm = get_llm(temperature)
    
    # Create memory with aggressive token optimization to avoid 413 errors
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=800,  # Reduced from 1500 to fit within Groq's 6000 token limit
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Restore chat history from session
    for i in range(0, len(session_history), 2):
        if i + 1 < len(session_history):
            memory.chat_memory.add_user_message(session_history[i])
            memory.chat_memory.add_ai_message(session_history[i + 1])
    
    # Create custom prompt
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"{config.SYSTEM_PROMPT}\n\nContext from documents:\n{{context}}\n\nQuestion: {{question}}\n\nAnswer:"
    )
    
    # Create retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_docs}
        ),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
        max_tokens_limit=config.LLM_MAX_TOKENS
    )
    
    return chain, memory


def get_memory_stats(memory) -> Dict:
    """Extract memory statistics"""
    try:
        buffer_messages = memory.chat_memory.messages if hasattr(memory.chat_memory, 'messages') else []
        buffer_count = len(buffer_messages)
        has_summary = hasattr(memory, 'moving_summary_buffer') and memory.moving_summary_buffer
        
        total_tokens = 0
        if buffer_messages:
            try:
                total_tokens = memory.llm.get_num_tokens(str(memory.load_memory_variables({})))
            except:
                pass
        
        return {
            "messages_in_memory": buffer_count,
            "has_summary": has_summary,
            "tokens_used": total_tokens
        }
    except Exception as e:
        print(f"Warning: Failed to get memory stats: {e}")
        return {"messages_in_memory": 0, "has_summary": False, "tokens_used": 0}


# ============================================
# API Endpoints
# ============================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("=" * 50)
    print("Starting RAG Chatbot API")
    print("=" * 50)
    try:
        load_vectorstore()
        print("✓ All models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vectorstore_loaded": _vectorstore is not None,
        "active_sessions": len(sessions)
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "vectorstore_loaded": _vectorstore is not None,
        "active_sessions": len(sessions)
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    response: Response,
    session_id: Optional[str] = Cookie(None)
):
    """
    Main chat endpoint - accepts prompt, returns context-aware answer
    
    - Automatically manages sessions via cookies
    - Maintains conversation context
    - Returns answer with source documents
    """
    try:
        # Load vectorstore
        vectorstore = load_vectorstore()
        
        # Get or create session
        session_id = get_or_create_session(session_id)
        
        # Set session cookie
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=SESSION_TIMEOUT,
            httponly=True,
            samesite="lax",
            secure=False  # Set to True in production with HTTPS
        )
        
        # Get session history
        history = get_session_history(session_id)
        
        # Use request params or defaults
        temperature = request.temperature if request.temperature is not None else config.LLM_TEMPERATURE
        k_docs = request.k_documents if request.k_documents is not None else config.TOP_K_DOCUMENTS
        
        # Create RAG chain with context
        chain, memory = create_rag_chain(vectorstore, temperature, k_docs, history)
        
        # Process query with retry logic for token limit errors
        try:
            result = chain.invoke({"question": request.prompt})
        except Exception as e:
            error_str = str(e)
            # Check if it's a token limit error
            if "413" in error_str or "rate_limit_exceeded" in error_str or "Request too large" in error_str:
                print(f"Token limit exceeded, retrying with reduced context...")
                # Retry with minimal context
                chain, memory = create_rag_chain(vectorstore, temperature, 1, [])  # Only 1 doc, no history
                result = chain.invoke({"question": request.prompt})
            else:
                raise
        
        answer = result["answer"]
        source_docs = result.get("source_documents", [])
        
        # Save to session
        save_to_session(session_id, request.prompt, answer)
        
        # Format sources - aggressively truncate to reduce token usage
        sources = []
        for doc in source_docs[:2]:  # Limit to first 2 sources only
            sources.append({
                "page": doc.metadata.get("page", 0),
                "content": doc.page_content[:150]  # Further reduced truncation
            })
        
        # Get memory stats
        memory_status = get_memory_stats(memory)
        
        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "memory_status": memory_status
        }
        
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/api/clear-session")
async def clear_session(
    response: Response,
    session_id: Optional[str] = Cookie(None)
):
    """Clear conversation history for current session"""
    if session_id and session_id in sessions:
        sessions[session_id]["history"] = []
        sessions[session_id]["last_access"] = time.time()
        return {"message": "Session cleared successfully", "session_id": session_id}
    
    return {"message": "No active session found"}


@app.get("/api/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "active_sessions": len(sessions),
        "vectorstore_loaded": _vectorstore is not None,
        "sessions": {
            sid: {
                "message_count": len(data["history"]),
                "last_access": datetime.fromtimestamp(data["last_access"]).isoformat(),
                "created_at": data.get("created_at", "unknown")
            }
            for sid, data in sessions.items()
        }
    }


# ============================================
# Run Server (for local development)
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 50)
    print("Starting FastAPI server on http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 50 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes
    )
