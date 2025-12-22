# Production-Ready RAG Chatbot

Context-aware, optimized RAG system for PDF documents using Groq LLM.

## ✨ Features

- **Context-Aware**: Maintains conversation history (last 5 exchanges)
- **Optimized Performance**: Cached embeddings and vector store loading
- **Production-Ready**: Error handling, validation, and backup mechanisms
- **Clean UI**: Streamlit interface with source citations
- **Configurable**: Centralized configuration in `config.py`

## 🏗️ Architecture

```
├── config.py              # Centralized configuration
├── create_vectordb.py     # Vector database creation (run once)
├── rag_chatbot.py        # CLI chatbot interface
├── app.py                # Streamlit web interface
└── .env                  # Environment variables
```

## 📦 Installation

```powershell
# Activate virtual environment
.\rag\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Step 1: Create Vector Database
```powershell
python create_vectordb.py
```

This will:
- Find your PDF file
- Create optimized chunks (800 chars, 150 overlap)
- Build FAISS vector store
- Validate and save with backup

### Step 2: Run Chatbot

**Web Interface (Recommended):**
```powershell
streamlit run app.py
```

**Terminal Interface:**
```powershell
python rag_chatbot.py
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
CHUNK_SIZE = 800              # Text chunk size
CHUNK_OVERLAP = 150           # Chunk overlap
TOP_K_DOCUMENTS = 3           # Context chunks to retrieve
LLM_TEMPERATURE = 0.2         # Response creativity (0-1)
LLM_MAX_TOKENS = 1024         # Maximum response length
```

## 🎯 Optimizations

1. **Context Awareness**: Uses `ConversationBufferWindowMemory` to maintain last 5 exchanges
2. **Smart Chunking**: Optimized chunk size (800) with semantic separators
3. **Caching**: Streamlit caches embeddings and vector store
4. **Efficient Retrieval**: Top-3 most relevant chunks (configurable)
5. **Error Handling**: Comprehensive validation and error messages
6. **Backup System**: Automatic backup of existing vector stores

## 📝 Commands (CLI)

- `quit` / `exit` - Exit chatbot
- `reset` - Clear conversation history
- `context` - Show current conversation context

## 🔧 Environment Variables

Required in `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## 📊 Performance Tips

- Reduce `TOP_K_DOCUMENTS` for faster responses
- Lower `LLM_TEMPERATURE` for more focused answers
- Increase `CHUNK_SIZE` for longer context per chunk

## 🛡️ Production Features

- ✅ Input validation
- ✅ Error recovery
- ✅ Automatic backups
- ✅ Context window management
- ✅ Source attribution
- ✅ Empty page filtering
- ✅ Small chunk filtering (>50 chars)

## 📄 License

MIT
