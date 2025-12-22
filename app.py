"""
Production-Ready Streamlit UI for RAG Chatbot
Optimized with caching, error handling, and context awareness
"""

import os
import hashlib
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from config import config

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streaming callback for Streamlit
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.complete = False
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)
    
    def on_llm_end(self, *args, **kwargs):
        self.complete = True


# Optimized CSS
st.markdown("""
<style>
    .stApp { max-width: 1400px; margin: 0 auto; }
    .source-box { 
        background-color: #f8f9fa; 
        color: #212529;
        padding: 12px; 
        border-radius: 5px; 
        margin: 8px 0;
        font-size: 0.9em;
        border-left: 3px solid #0066cc;
    }
    .source-box code {
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    """Load vector store with caching for performance."""
    if not os.path.exists(config.VECTORDB_PATH):
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = FAISS.load_local(
            config.VECTORDB_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")
        return None


@st.cache_resource(show_spinner=False)
def get_llm(_temperature, _streaming=False):
    """Get LLM instance with caching and optional streaming."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("❌ GROQ_API_KEY not found!")
        st.stop()
    
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=config.LLM_MODEL,
        temperature=_temperature,
        max_tokens=config.LLM_MAX_TOKENS,
        streaming=_streaming
    )


def initialize_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory_status" not in st.session_state:
        st.session_state.memory_status = {"full_detail": 0, "has_summary": False, "total_tokens": 0}


def get_memory_stats(memory):
    """Extract memory statistics for display."""
    try:
        # Get buffer messages (recent, full detail)
        buffer_messages = memory.chat_memory.messages if hasattr(memory.chat_memory, 'messages') else []
        buffer_count = len(buffer_messages)
        
        # Check if there's a moving summary (old messages)
        has_summary = hasattr(memory, 'moving_summary_buffer') and memory.moving_summary_buffer
        
        # Estimate tokens
        total_tokens = memory.llm.get_num_tokens(str(memory.load_memory_variables({}))) if buffer_messages else 0
        
        return {
            "full_detail": buffer_count,
            "has_summary": has_summary,
            "total_tokens": total_tokens
        }
    except Exception as e:
        print(f"Warning: Failed to get memory stats: {e}")
        return {"full_detail": 0, "has_summary": False, "total_tokens": 0}


def create_chain(vectorstore, temperature, k_docs, streaming=False, callbacks=None):
    """Create optimized conversational chain with smart context management."""
    llm = get_llm(temperature, streaming)
    
    # Smart memory: keeps recent messages + summarizes older ones to save tokens
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1500,  # Only use ~1500 tokens for history
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Load existing chat history into memory
    for i in range(0, len(st.session_state.chat_history), 2):
        if i + 1 < len(st.session_state.chat_history):
            memory.chat_memory.add_user_message(st.session_state.chat_history[i])
            memory.chat_memory.add_ai_message(st.session_state.chat_history[i + 1])
    
    # Context-aware prompt template with optimized history
    qa_prompt = PromptTemplate(
        template=config.SYSTEM_PROMPT + """

Document context:
{context}

Relevant conversation context (recent + summary of older messages):
{chat_history}

Current question: {question}

Answer using the document context and conversation history:""",
        input_variables=["context", "chat_history", "question"]
    )
    
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
    
    return chain


def display_sources(sources):
    """Display source documents in a clean format with proper visibility."""
    if not sources:
        return
    
    with st.expander("📖 View Sources", expanded=False):
        for i, source in enumerate(sources[:3], 1):  # Limit to top 3 sources
            page_num = source.metadata.get('page', 'N/A')
            if isinstance(page_num, int):
                page_num += 1
            
            st.markdown(f"**Source {i}** (Page {page_num})")
            content = source.page_content[:400].replace('\n', ' ')
            # Use HTML with explicit text color
            st.markdown(
                f'<div class="source-box" style="color: #212529;">{content}...</div>', 
                unsafe_allow_html=True
            )


def main():
    st.title("🤖 RAG Document Chatbot")
    st.caption("Context-aware AI assistant for your documents")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check vector store
        if not os.path.exists(config.VECTORDB_PATH):
            st.error("❌ Vector store not found!")
            st.warning("Run `python create_vectordb.py` first")
            st.stop()
        
        st.success("✅ Vector store loaded")
        
        # Settings
        temperature = st.slider(
            "Temperature", 
            0.0, 1.0, 
            float(config.LLM_TEMPERATURE), 
            0.05,
            help="Lower = more focused, Higher = more creative"
        )
        
        k_documents = st.slider(
            "Context chunks", 
            1, 5, 
            config.TOP_K_DOCUMENTS,
            help="Number of document chunks to retrieve"
        )
        
        enable_streaming = st.checkbox(
            "⚡ Streaming responses",
            value=True,
            help="Show response as it's generated (faster perceived speed)"
        )
        
        st.divider()
        
        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("ℹ️ Info", use_container_width=True):
                st.info(f"""
                **Model:** {config.LLM_MODEL}
                **Chunks:** {config.TOP_K_DOCUMENTS}
                **Context:** Smart history (~1500 tokens)
                **Memory:** Recent + summary of old
                """)
        
        st.divider()
        
        # Show conversation stats
        user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
        st.caption(f"💬 Total Messages: {len(st.session_state.messages)}")
        st.caption(f"📝 Questions: {len(user_messages)}")
        st.caption("⚡ Token-optimized context")
        
        # Memory status indicator
        if st.session_state.get("memory_status"):
            mem_status = st.session_state.memory_status
            
            st.divider()
            st.markdown("**🧠 Memory Status**")
            
            if mem_status.get("has_summary"):
                st.markdown("""
                <div style='background-color: #e3f2fd; color: #1565c0; padding: 10px; border-radius: 5px; font-size: 0.85em;'>
                    📦 <b>Summarized:</b> Older messages<br>
                    📄 <b>Full Detail:</b> {} recent messages<br>
                    🎯 <b>Tokens:</b> ~{} / 1500
                </div>
                """.format(
                    mem_status.get("full_detail", 0),
                    mem_status.get("total_tokens", 0)
                ), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background-color: #f1f8e9; color: #2e7d32; padding: 10px; border-radius: 5px; font-size: 0.85em;'>
                    ✅ <b>All messages in full detail</b><br>
                    🎯 <b>Tokens:</b> ~{} / 1500
                </div>
                """.format(mem_status.get("total_tokens", 0)), unsafe_allow_html=True)
    
    # Load vector store
    with st.spinner("🔄 Initializing..."):
        vectorstore = load_vectorstore()
    
    if vectorstore is None:
        st.error("❌ Failed to load vector store")
        st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                display_sources(message.get("sources", []))
    
    # Chat input
    if prompt := st.chat_input("Ask about the document...", key="chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            try:
                if enable_streaming:
                    # Streaming mode - show tokens as they generate
                    response_container = st.empty()
                    stream_handler = StreamHandler(response_container)
                    
                    chain = create_chain(vectorstore, temperature, k_documents, streaming=True)
                    # Pass callbacks to invoke, not chain creation
                    result = chain.invoke(
                        {"question": prompt},
                        {"callbacks": [stream_handler]}
                    )
                    # Use streamed text if available, otherwise fallback
                    answer = stream_handler.text if stream_handler.complete else result["answer"]
                    sources = result.get("source_documents", [])
                    
                    # Ensure final answer is displayed
                    response_container.markdown(answer)
                    
                else:
                    # Non-streaming mode
                    with st.spinner("🤔 Thinking..."):
                        chain = create_chain(vectorstore, temperature, k_documents)
                        result = chain.invoke({"question": prompt})
                        answer = result["answer"]
                        sources = result.get("source_documents", [])
                        
                        # Display answer
                        st.markdown(answer)
                
                # Update memory status (common for both modes)
                st.session_state.memory_status = get_memory_stats(chain.memory)
                st.session_state.chat_history.append(prompt)
                st.session_state.chat_history.append(answer)
                
                # Display sources
                display_sources(sources)
                
                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
