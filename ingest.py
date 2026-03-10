import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "test data"
CHROMA_DB_PATH = "./chroma_db"

def build_vector_database():
    # Validate data directory exists before proceeding
    if not os.path.exists(DATA_DIR):
        print(f" Error: The folder '{DATA_DIR}' does not exist.")
        print("Please create it and add your PDF files, then run this script again.")
        return

    print(f" Loading PDF documents from '{DATA_DIR}' directory...")

    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()

    if not documents:
        print(f" Error: No PDF files found in '{DATA_DIR}'. Please add your files and try again.")
        return

    print(f" Successfully loaded {len(documents)} pages. Splitting text into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"  Created {len(chunks)} text chunks.")

    print(" Generating embeddings and building ChromaDB...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(CHROMA_DB_PATH):
        print(
            f"  Warning: A ChromaDB already exists at '{CHROMA_DB_PATH}'.\n"
            "This will ADD new documents on top of existing ones.\n"
            "If you want a clean rebuild, delete the 'chroma_db' folder first."
        )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    print(
        f"\n Vector Database successfully created!\n"
        f" Stats: {len(documents)} pages → {len(chunks)} chunks stored in '{CHROMA_DB_PATH}'"
    )

if __name__ == "__main__":
    build_vector_database()
