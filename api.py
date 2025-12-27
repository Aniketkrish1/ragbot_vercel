# ... (Imports)
import os
import uuid
import time
import functools
import json
from typing import Optional, Dict, List, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Cookie, Response, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from config import config

# Load environment variables
load_dotenv()

# Fix OpenMP library conflict warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Stateless Context-aware PDF Question Answering with Groq LLM",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for loaded models
_vectorstore = None
_embeddings = None
_llm = None


# ============================================
# Custom HuggingFace API Embeddings Wrapper
# ============================================

class HuggingFaceAPIEmbeddings(Embeddings):
    """Custom embeddings class using HuggingFace Inference API"""
    
    def __init__(self, model_name: str, api_key: str):
        from huggingface_hub import InferenceClient
        import numpy as np
        
        self.model_name = model_name
        self.api_key = api_key
        self.client = InferenceClient(token=api_key)
        self.np = np
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents"""
        embeddings = []
        for text in texts:
            embedding = self._embed_single(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query"""
        return self._embed_single(text)
    
    def _embed_single(self, text: str) -> list[float]:
        """Helper method to embed a single text using InferenceClient"""
        import time
        
        max_retries = 3
        retry_attempt = 0
        
        while retry_attempt < max_retries:
            try:
                # Use feature_extraction which works with the router endpoint
                result = self.client.feature_extraction(
                    text=text,
                    model=self.model_name
                )
                
                # Parse the result
                embedding = self._parse_embedding(result)
                
                # Validate dimensions
                if len(embedding) == 0:
                    raise ValueError("Received empty embedding vector")
                
                return embedding
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle model loading
                if "loading" in error_str or "503" in error_str:
                    print(f"Model loading, waiting 20s...")
                    time.sleep(20)
                    continue  # Don't increment retry
                
                # Handle rate limiting
                if "429" in error_str or "rate" in error_str:
                    wait_time = min(2 ** retry_attempt, 30)
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retry_attempt += 1
                    continue
                
                # Other errors
                retry_attempt += 1
                if retry_attempt >= max_retries:
                    raise RuntimeError(
                        f"HuggingFace API error after {max_retries} attempts: {str(e)}"
                    )
                time.sleep(2 ** retry_attempt)
        
        raise RuntimeError("Failed to get embeddings from HuggingFace API")
    
    def _parse_embedding(self, result) -> list[float]:
        """Parse API response to extract embedding vector"""
        # Handle numpy array
        if isinstance(result, self.np.ndarray):
            # If 2D array, flatten or take first row
            if result.ndim == 2:
                return result[0].tolist() if result.shape[0] > 0 else result.flatten().tolist()
            return result.tolist()
        
        # Handle list formats
        if isinstance(result, list):
            if len(result) == 0:
                return []
            
            # Nested list [[embedding]] -> [embedding]
            if isinstance(result[0], list):
                return result[0]
            
            # Already flat list [embedding]
            if isinstance(result[0], (int, float)):
                return result
        
        # Fallback: try to convert to list
        return list(result) if result else []


# ============================================
# Request/Response Models
# ============================================


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="User question")
    user_id: str = Field(..., description="Unique User ID from frontend")
    session_id: Optional[str] = Field(default=None, description="Check for Stateless Session")
    history: List[str] = Field(default=[], description="REQUIRED: Chat history [Human, AI, Human, AI...]")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="LLM temperature")
    k_documents: Optional[int] = Field(default=None, ge=1, le=10, description="Number of context documents")


class SourceDocument(BaseModel):
    page: int
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    session_id: str


class HealthResponse(BaseModel):
    status: str
    vectorstore_loaded: bool


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
        
        embeddings = HuggingFaceAPIEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            api_key=hf_token
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
    """
    Create sophisticated RAG chain with:
    1. Query Rewriting (History Aware Retriever)
    2. MMR Search (Maximal Marginal Relevance) for better context
    3. Proper Answer Generation
    """
    
    # Trim history to keep it manageable (Last 6 messages = 3 turns)
    estimated_history_tokens = estimate_tokens(" ".join(session_history))
    if estimated_history_tokens > 2000:
        session_history = session_history[-6:]
    
    # Initialize LLM
    llm = get_llm(temperature)
    
    # 1. Define the Retriever with MMR (Better diversity than standard similarity)
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": k_docs,
            "lambda_mult": 0.7 
        }
    )
    
    # 2. Memory / History Management
    from langchain_core.messages import HumanMessage, AIMessage
    chat_history = []
    
    # Simple parser: Assumes strings come in as Raw text, or formatted "Human: ..."
    # We'll just take them as alternating turns if not prefixed, or parse prefix if present.
    # For robustness in this simple version, we assume the frontend sends simple strings 
    # OR we just blindly alternate. The safest for custom history arrays is blindly alternate 
    # assuming Trusted Frontend.
    
    # Logic: Even indices are Human, Odd are AI
    for i in range(len(session_history)):
        msg = session_history[i]
        # Cleanup prefixes if frontend sends "Human: hello"
        clean_msg = msg.replace("Human: ", "").replace("AI: ", "")
        
        if i % 2 == 0:
            chat_history.append(HumanMessage(content=clean_msg))
        else:
            chat_history.append(AIMessage(content=clean_msg))
    
    # 3. Create History-Aware Retriever (The "Rewrite" Step)
    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # 4. Create QA Chain (The "Answer" Step)
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    
    qa_system_prompt = (
        f"{config.SYSTEM_PROMPT}\n\n"
        "Context from documents:\n{context}\n\n"
        "If the answer is not in the context, say you don't know."
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain, chat_history


# ============================================
# API Endpoints
# ============================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("=" * 50)
    print("Starting RAG Chatbot API (Stateless)")
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
        "vectorstore_loaded": _vectorstore is not None
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "vectorstore_loaded": _vectorstore is not None
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless Chat Endpoint
    - Accepts `history` list (managed by frontend).
    - Returns answer based on docs + history context.
    - No DB persistence on this server.
    """
    try:
        # Load vectorstore
        vectorstore = load_vectorstore()
        
        # Use history directly from request
        history = request.history
        
        # Use request params or defaults
        temperature = request.temperature if request.temperature is not None else config.LLM_TEMPERATURE
        k_docs = request.k_documents if request.k_documents is not None else config.TOP_K_DOCUMENTS
        
        # Create RAG chain with context
        rag_chain, chat_history_objs = create_rag_chain(vectorstore, temperature, k_docs, history)
        
        # Process query
        try:
            result = rag_chain.invoke({
                "input": request.prompt,
                "chat_history": chat_history_objs
            })
        except Exception as e:
            error_str = str(e)
            # Fallback for minor token issues
            if "413" in error_str or "rate_limit_exceeded" in error_str:
                print(f"Warning: Token limit hit, retrying with zero context...")
                rag_chain_min, _ = create_rag_chain(vectorstore, temperature, 1, [])
                result = rag_chain_min.invoke({
                    "input": request.prompt,
                    "chat_history": []
                })
            else:
                raise
        
        answer = result["answer"]
        source_docs = result.get("context", [])
        
        # Format sources
        sources = []
        for doc in source_docs[:2]:
            sources.append({
                "page": doc.metadata.get("page", 0),
                "content": doc.page_content[:150]
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "session_id": "stateless",
        }
        
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("Starting FastAPI server on http://localhost:8000")
    print("=" * 50 + "\n")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
