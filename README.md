# Production-Ready RAG Chatbot

Context-aware, optimized RAG system for PDF documents using Groq LLM. **Ready for Vercel deployment!**

## ✨ Features

- **Context-Aware**: Maintains conversation history with Vercel KV session storage
- **Optimized Performance**: Cached embeddings and vector store loading
- **Production-Ready**: Deployed on Vercel serverless
- **Clean API**: FastAPI backend with cookie-based sessions
- **Configurable**: Centralized configuration in `config.py`

## 🚀 Quick Deploy to Vercel

```bash
# 1. Push to GitHub
git add .
git commit -m "ready for vercel deployment"
git push origin main

# 2. Go to vercel.com → Import GitHub repo
# 3. Add env vars: GROQ_API_KEY, HF_TOKEN
# 4. Deploy! (auto-configures Vercel KV)
```

See [DEPLOY-VERCEL.md](DEPLOY-VERCEL.md) for detailed instructions.

## 🏗️ Project Structure

```
├── api.py                 # FastAPI backend (main entry point)
├── config.py              # Centralized configuration
├── create_vectordb.py     # Vector database creation (run once)
├── rag_chatbot.py        # CLI chatbot interface
├── requirements-api.txt   # API dependencies (for Vercel)
├── vercel.json           # Vercel configuration
└── .env                  # Environment variables
```

## 📦 Local Development

```powershell
# Activate virtual environment
.\rag\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements-api.txt
```

## 🔌 API Endpoints

**Base URL:** `https://your-project.vercel.app`

### Health Check
```bash
GET /health
```

### Chat
```bash
POST /api/chat
Content-Type: application/json

{
  "prompt": "Your question here"
}
```

Response includes answer, sources, and session ID.

## 🚀 Local Testing

### 1. Create Vector Database
```powershell
python create_vectordb.py
```

### 2. Run API Locally
```powershell
# Start dev server
uvicorn api:app --reload
```

Visit: `http://localhost:8000/docs` for interactive API docs

### 3. Test Chat Endpoint
```powershell
curl -X POST http://localhost:8000/api/chat `
  -H "Content-Type: application/json" `
  -d '{"prompt": "What is leadership?"}'
```

### CLI Testing (Original)
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
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

**Vercel Setup:** Add these in project **Settings → Environment Variables**

## 📊 Performance Tips

- Reduce `TOP_K_DOCUMENTS` for faster responses
- Lower `LLM_TEMPERATURE` for more focused answers
- Increase `CHUNK_SIZE` for longer context per chunk

## 🛡️ Deployment Features

**On Vercel:**
- ✅ Serverless FastAPI backend
- ✅ Vercel KV for session storage (Redis)
- ✅ Cookie-based session management
- ✅ Automatic HTTPS
- ✅ Cold start < 2 seconds
- ✅ Auto-scaling (10K free KV commands/month)

**Local Development:**
- ✅ In-memory session fallback
- ✅ Full compatibility
- ✅ Interactive API docs at `/docs`

## 📄 License

MIT
