import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv

#Load API Key from .env file 
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found! Please create a .env file with: GROQ_API_KEY=your_key_here"
    )
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

#2: Check if ChromaDB exists before loading (prevents crash if ingest.py wasn't run)
CHROMA_DB_PATH = "./chroma_db"
if not os.path.exists(CHROMA_DB_PATH):
    raise FileNotFoundError(
        "ChromaDB not found! Please run ingest.py first to build the vector database."
    )

# 1. Load local free Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Connect to the ChromaDB vector database
vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

retriever = vector_db.as_retriever(search_kwargs={"k": 6})

# 3. Initialize Groq clients
groq_client = Groq(api_key=GROQ_API_KEY)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    streaming=True
)

# 4. Build the Prompt
system_prompt = (
    "You are Mentallico, an empathetic, professional, and highly knowledgeable AI psychiatrist. "
    "Use the following extracted information from the medical documents to answer the patient's question. "
    "Synthesize the information into a natural, conversational, compassionate, and helpful response. "
    "Do NOT just copy-paste the excerpts. "
    "If you do not know the answer based on the provided context, politely and professionally state that you do not have enough information and do not guess.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Previous Conversation History:\n{chat_history}\n\nPatient's New Message: {input}"),
])

# 5. Helper to rebuild the RAG chain (needed after adding new PDFs)
def _build_rag_chain():
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

rag_chain = _build_rag_chain()

# 6. Audio Transcription Function
def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_file_path, file.read()),
                model="whisper-large-v3"
            )
        return transcription.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def generate_answer(user_query, history_list=None):
    """
    Generates a streaming answer from the RAG chain.
    Yields response tokens one by one for real-time display in Gradio.
    """
    if history_list is None:
        history_list = []

    try:
        MAX_HISTORY = 10
        recent_history = history_list[-MAX_HISTORY:]

        chat_history_str = ""
        for msg in recent_history:
            if msg[0] and msg[1]:  # safety check: skip incomplete messages
                chat_history_str += f"Patient: {msg[0]}\nMentallico: {msg[1]}\n"

        full_response = ""
        for chunk in rag_chain.stream({
            "input": user_query,
            "chat_history": chat_history_str
        }):
            # The answer key arrives in chunks
            if "answer" in chunk:
                token = chunk["answer"]
                full_response += token
                yield full_response  # yield partial response for Gradio streaming

    except Exception as e:
        yield f"An error occurred while connecting to the model. Error: {str(e)}"


# 8. Process New PDF Function
def process_new_pdf(file_path):
    """
    Loads a new PDF, splits it, adds it to the vector DB,
    and refreshes the retriever & RAG chain so new content is immediately queryable.
    """
    global retriever, rag_chain

    try:
        if file_path is None:
            return "Please select a file first."

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            return "The PDF appears to be empty or unreadable."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # Add to vector DB
        vector_db.add_documents(chunks)

        # Refresh retriever & RAG chain so new docs are included in future queries
        retriever = vector_db.as_retriever(search_kwargs={"k": 6})
        rag_chain = _build_rag_chain()

        return (
            f" Successfully processed '{os.path.basename(file_path)}'.\n"
            f" {len(docs)} pages → {len(chunks)} chunks added to the knowledge base.\n"
            f" Knowledge base updated and ready!"
        )

    except Exception as e:
        return f" Error processing file: {str(e)}"
