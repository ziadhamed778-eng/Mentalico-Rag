# Mentallico AI - Psychiatric RAG Engine

This is a Modular Retrieval-Augmented Generation (RAG) system built with LangChain, designed specifically as the core engine for **Mentallico**, an AI-powered psychiatric diagnostic assistant. The project reads medical PDF files, processes them, and allows users to query the content using a highly advanced LLM via a modern Gradio interface.

## Project Architecture (1-Page Summary)

### 1. Vector Database: ChromaDB
**Reason for selection:** ChromaDB is an open-source, lightweight, and locally-running vector database. It integrates seamlessly with LangChain and ensures that all sensitive medical documents and context remain secure locally without relying on external cloud databases.

### 2. Large Language Model (LLM): Llama-3.3-70B (via Groq API)
**Reason for selection:** We transitioned from local execution to Groq's API utilizing the massive `llama-3.3-70b-versatile` model. This architectural shift provides three major advantages for a medical/psychiatric application:
- **Advanced Reasoning:** The 70-billion parameter model offers deep, empathetic, and highly accurate clinical reasoning that smaller local models cannot match.
- **Ultra-Low Latency:** Groq's LPU inference engine delivers responses in milliseconds, ensuring a seamless conversational experience for the user.
- **Hardware Independence:** Offloading the heavy LLM computation allows the system to run on any machine without requiring high-end local GPUs or suffering from local networking (WSL) bottlenecks.

### 3. Embeddings: HuggingFace (`all-MiniLM-L6-v2`)
**Reason for selection:** Highly optimized for semantic search and runs completely locally without any API costs, providing an excellent balance between performance, speed, and accuracy for vectorizing medical text.

---

## How to Run the Project Locally

To test the Mentallico RAG system, please follow these steps carefully:

### Prerequisites:
1. You need a free API key from [Groq Cloud](https://console.groq.com/).
2. Python 3.8+ installed on your machine.

### Setup Instructions:
1. Clone this repository.
2. Create and activate a virtual environment:
   - Linux/Mac/WSL: `python -m venv venv` and `source venv/bin/activate`
   - Windows: `python -m venv venv` and `venv\Scripts\activate`
3. Install the required dependencies:
   `pip install -r requirements.txt`
4. Set your Groq API Key:
   - Ensure your API key is placed inside the `rag_engine.py` file, or export it in your terminal:
     `export GROQ_API_KEY="gsk_your_api_key_here"`
5. Place your medical/psychiatric test PDF files inside the designated data folder.

### Running the Application:
1. **Ingest the Data:** Run the ingestion script to vectorize the PDFs and create the Vector Database.
   `python ingest.py`
2. **Launch the UI:** Start the Mentallico Gradio interface.
   `python app.py`
3. Open the provided local URL (e.g., `http://127.0.0.1:7860`) in your browser and start consulting the AI psychiatrist!