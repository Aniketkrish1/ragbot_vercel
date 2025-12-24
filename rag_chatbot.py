"""
Production-Ready RAG Chatbot with Context Awareness
Optimized for performance and reliability
"""

import os
import sys
from typing import Dict, Optional
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from config import config

load_dotenv()


class RAGChatbot:
    """Context-aware RAG chatbot with optimized performance."""
    
    def __init__(self, vectordb_path: str = None):
        """Initialize the RAG chatbot with pre-built vector store."""
        self.vectordb_path = vectordb_path or config.VECTORDB_PATH
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        if not os.path.exists(self.vectordb_path):
            raise FileNotFoundError(
                f"Vector store not found at '{self.vectordb_path}'. "
                "Run 'python create_vectordb.py' first."
            )
        
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.chain = None
        self.memory = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize all components with error handling."""
        try:
            print("🔄 Loading vector store...")
            self._load_vectorstore()
            
            print("🤖 Initializing LLM with context-aware prompts...")
            self._setup_llm_chain()
            
            print("✅ RAG Chatbot ready!\n")
            
        except Exception as e:
            raise RuntimeError(f"Initialization failed: {str(e)}")
    
    def _load_vectorstore(self):
        """Load the pre-built FAISS vector store."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = FAISS.load_local(
            self.vectordb_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def _setup_llm_chain(self):
        """Set up LLM with context-aware conversation chain."""
        # Initialize Groq LLM with optimized settings
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
        
        # Use window memory with reduced context (last 3 exchanges) to avoid token limits
        self.memory = ConversationBufferWindowMemory(
            k=3,  # Reduced from 5 to prevent exceeding Groq's 6000 token limit
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Custom prompt for context-aware responses
        qa_prompt = PromptTemplate(
            template=config.SYSTEM_PROMPT + """

Context from document:
{context}

Conversation history:
{chat_history}

Current question: {question}

Provide a clear, accurate answer based on the context. If referring to previous questions, acknowledge them explicitly.""",
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create conversational retrieval chain with optimizations
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": config.TOP_K_DOCUMENTS}
            ),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=False,
            max_tokens_limit=config.LLM_MAX_TOKENS
        )
    
    def chat(self, query: str) -> Dict:
        """
        Process a query with full context awareness.
        
        Args:
            query: User's question
            
        Returns:
            Dict with 'answer' and 'sources'
        """
        if not query or not query.strip():
            return {"answer": "Please provide a valid question.", "sources": []}
        
        try:
            result = self.chain.invoke({"question": query.strip()})
            return {
                "answer": result["answer"],
                "sources": result.get("source_documents", [])
            }
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": []
            }
    
    def reset_memory(self):
        """Clear conversation history."""
        self.memory.clear()
    
    def get_context_summary(self) -> str:
        """Get summary of current conversation context."""
        return self.memory.load_memory_variables({}).get("chat_history", "No context yet")


def main():
    """CLI interface for the chatbot."""
    print("\n" + "="*60)
    print("📚 RAG Chatbot - Production Mode")
    print("="*60)
    
    try:
        chatbot = RAGChatbot()
        
        print("\nCommands:")
        print("  'quit' or 'exit' - Exit the chatbot")
        print("  'reset' - Clear conversation history")
        print("  'context' - Show conversation context")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'reset':
                    chatbot.reset_memory()
                    print("🔄 Conversation history cleared\n")
                    continue
                
                if user_input.lower() == 'context':
                    print(f"\n📝 Context: {chatbot.get_context_summary()}\n")
                    continue
                
                # Get response
                response = chatbot.chat(user_input)
                print(f"\n🤖 Assistant: {response['answer']}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
                
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
