"""
Optimized Vector Database Creation Script
Production-ready with error handling and validation
"""

import os
import sys
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from config import config

load_dotenv()


def validate_pdf_exists() -> str:
    """Find and validate PDF file existence."""
    pdf_files = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the current directory")
    
    if len(pdf_files) > 1:
        print(f"⚠️  Found {len(pdf_files)} PDF files. Using: {pdf_files[0]}")
    
    return pdf_files[0]


def load_and_validate_pdf(pdf_path: str) -> List[Document]:
    """Load PDF and validate content."""
    print(f"📄 Loading PDF: {pdf_path}")
    
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("PDF is empty or couldn't be read")
        
        # Filter out empty pages
        documents = [doc for doc in documents if doc.page_content.strip()]
        
        print(f"✅ Loaded {len(documents)} pages with content")
        return documents
        
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {str(e)}")


def create_optimized_chunks(documents: List[Document]) -> List[Document]:
    """Split documents with optimized parameters."""
    print(f"✂️  Chunking (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Filter out very small chunks
    chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 50]
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def create_vectorstore_with_validation(chunks: List[Document]) -> FAISS:
    """Create and validate vector store."""
    print(f"🧮 Loading embeddings: {config.EMBEDDING_MODEL}")
    
    if config.USE_HF_INFERENCE_API:
        # Use HuggingFace Inference API (free tier, no local storage)
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("\n❌ Error: HF_TOKEN not found in environment variables")
            print("Get your free token at: https://huggingface.co/settings/tokens")
            print("Add to .env file: HF_TOKEN=your_token_here")
            print("\nOr set USE_HF_INFERENCE_API=False in config.py to use local model")
            raise RuntimeError("HF_TOKEN required when USE_HF_INFERENCE_API=True")
        
        embeddings = HuggingFaceEndpointEmbeddings(
            model=config.EMBEDDING_MODEL,
            huggingfacehub_api_token=hf_token
        )
        print("✓ Using HuggingFace Inference API (free tier, no local storage needed)")
    else:
        # Use local model (requires ~80-90MB storage)
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ Using local embeddings model (~80-90MB)")
    
    print("📦 Creating FAISS vector store...")
    
    # Create vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Validate vector store
    test_results = vectorstore.similarity_search("test", k=1)
    if not test_results:
        raise RuntimeError("Vector store validation failed")
    
    print("✅ Vector store created and validated")
    return vectorstore


def save_with_backup(vectorstore: FAISS, path: str):
    """Save vector store with backup of existing."""
    if os.path.exists(path):
        backup_path = f"{path}.backup"
        if os.path.exists(backup_path):
            import shutil
            shutil.rmtree(backup_path)
        os.rename(path, backup_path)
        print(f"📋 Backed up existing vector store to: {backup_path}")
    
    vectorstore.save_local(path)
    print(f"💾 Saved vector store to: {path}")


def main():
    """Main execution with error handling."""
    print("\n" + "="*60)
    print("🚀 RAG Vector Database Creation - Production Mode")
    print("="*60 + "\n")
    
    try:
        # Step 1: Validate PDF
        pdf_path = validate_pdf_exists()
        
        # Step 2: Load and validate PDF
        documents = load_and_validate_pdf(pdf_path)
        
        # Step 3: Create optimized chunks
        chunks = create_optimized_chunks(documents)
        
        # Step 4: Create and validate vector store
        vectorstore = create_vectorstore_with_validation(chunks)
        
        # Step 5: Save with backup
        save_with_backup(vectorstore, config.VECTORDB_PATH)
        
        print("\n" + "="*60)
        print("✅ Vector database created successfully!")
        print(f"📁 Location: {os.path.abspath(config.VECTORDB_PATH)}")
        print(f"📊 Total chunks: {len(chunks)}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
