import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_database():
    print(" Loading PDF documents from 'test data' directory")
    
    # 1. Load PDFs from the specific directory required by the assignment
    loader = PyPDFDirectoryLoader("test data")
    documents = loader.load()

    if not documents:
        print(" Error: No PDF files found in the 'test data' folder. Please add your files and try again.")
        return

    print(f" Successfully loaded {len(documents)} pages. Splitting text into chunks...")
    
    # 2. Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(" Generating embeddings and building ChromaDB...")
    
    # 3. Use local HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Create and persist the vector store
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print(" Vector Database successfully created!")

if __name__ == "__main__":
    build_vector_database()
