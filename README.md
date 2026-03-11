# Mentallico AI - Psychiatric RAG Engine

This is a Modular Retrieval-Augmented Generation (RAG) system built with the latest **LangChain (LCEL)** architecture, designed specifically as the core engine for **Mentallico**, an AI-powered psychiatric diagnostic assistant. The project reads medical PDF files, processes them, and allows users to query the content using a highly advanced LLM via a modern **Gradio 6** interface with both **Text and Voice** capabilities.

## Project Architecture (1-Page Summary)

### 1. Vector Database: ChromaDB
**Reason for selection:** ChromaDB is an open-source, lightweight, and locally-running vector database. It integrates seamlessly with LangChain and ensures that all sensitive medical documents and context remain secure locally without relying on external cloud databases.

### 2. Large Language Model (LLM): Llama-3.3-70B (via Groq API)
**Reason for selection:** We utilize Groq's API with the massive `llama-3.3-70b-versatile` model. This architectural shift provides three major advantages for a medical/psychiatric application:
- **Advanced Reasoning:** The 70-billion parameter model offers deep, empathetic, and highly accurate clinical reasoning that smaller local models cannot match.
- **Ultra-Low Latency & Streaming:** Groq's LPU inference engine delivers responses in milliseconds, allowing for real-time text streaming, ensuring a seamless conversational experience.
- **Bilingual Empathy:** The model is specifically prompted to understand and respond dynamically in both Arabic and English, maintaining a professional psychiatric persona.

### 3. Audio Processing: Whisper Large V3 (via Groq API)
**Reason for selection:** To enhance accessibility and simulate a real clinic environment, users can speak their symptoms. We integrated `whisper-large-v3` for lightning-fast, highly accurate multilingual voice-to-text transcription.

### 4. Embeddings: HuggingFace (`all-MiniLM-L6-v2`)
**Reason for selection:** Highly optimized for semantic search and runs completely locally without any API costs, providing an excellent balance between performance, speed, and accuracy for vectorizing medical text.

### 5. RAG Pipeline: LCEL (LangChain Expression Language)
**Reason for selection:** Upgraded from legacy chains to LCEL to ensure a robust, state-of-the-art data pipeline that flawlessly handles complex chat dictionaries, formatting, and strictly prevents "list object" replacement bugs.

---

## How to Run the Project Locally

To test the Mentallico RAG system, please follow these steps carefully:

### Prerequisites:
1. You need a free API key from [Groq Cloud](https://console.groq.com/).
2. Python 3.8+.

### Setup Instructions:
1. Clone this repository.
2. Create and activate a virtual environment:
   - Linux/Mac/WSL: `python -m venv venv` and `source venv/bin/activate`
   - Windows: `python -m venv venv` and `venv\Scripts\activate`
3. Install the required dependencies (Ensuring LangChain and Gradio 6 compatibility):
   `pip install -r requirements.txt`
4. Set your Groq API Key using a `.env` file:
   - Create a file named `.env` in the root directory.
   - Add the following line exactly as shown (no quotes or spaces):
     `GROQ_API_KEY=gsk_your_api_key_here`

### Running the Application:
1. **Ingest Initial Data (Optional):** Run the ingestion script to vectorize any initial PDFs in your data folder and build the database.
   `python ingest.py`
2. **Launch the UI:** Start the Mentallico interface.
   `python app.py`
3. Open the provided local URL (e.g., `http://127.0.0.1:7860`) in your browser.

### Features to Try in the UI:
- **Text Chat:** Ask complex medical questions based on the uploaded literature.
- **Voice Chat 🎙️:** Use the microphone to speak your symptoms directly to the AI in English or Arabic.
- **Dynamic Upload 📥:** Upload new medical records or research papers directly from the UI to instantly expand Mentallico's knowledge base on the fly!
