import os
from operator import itemgetter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv

# 1. Load API Key from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found! Please check your .env file.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 2. Check if ChromaDB exists
CHROMA_DB_PATH = "./chroma_db"
if not os.path.exists(CHROMA_DB_PATH):
    raise FileNotFoundError("ChromaDB not found! Please run ingest.py first.")

# 3. Load Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Connect to Vector DB
vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 6})

# 5. Initialize Groq
groq_client = Groq(api_key=GROQ_API_KEY)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, streaming=True)

# 6. Build Prompt
system_prompt = (
    "You are Mentallico, an empathetic, professional, and highly knowledgeable AI psychiatrist. "
    "Use the following extracted information from the medical documents to answer the patient's question. "
    "Synthesize the information into a natural, conversational, compassionate, and helpful response. "
    "Do NOT just copy-paste the excerpts. "
    "If you do not know the answer based on the provided context, politely and professionally state that you do not have enough information and do not guess.\n\n"
    "IMPORTANT: Always respond in the same language the patient used. "
    "If they wrote in Arabic, reply in Arabic. If in English, reply in English.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Previous Conversation History:\n{chat_history}\n\nPatient's New Message: {input}"),
])

# 7. Helper — converts retrieved docs to plain text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 8. Build RAG chain using LCEL
def _build_rag_chain(ret=None):
    active_retriever = ret or retriever
    return (
        {
            "context":      itemgetter("input") | active_retriever | format_docs,
            "input":        itemgetter("input"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

rag_chain = _build_rag_chain()

# 9. Audio Transcription
def transcribe_audio(audio_file_path):
    try:
        if not audio_file_path:
            return ""
        with open(audio_file_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_file_path, file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
        return transcription
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

# 10. Generate Answer with Streaming
def generate_answer(user_query, history_list=None):
    if history_list is None:
        history_list = []

    try:
        user_query = str(user_query)

        MAX_HISTORY = 10
        recent_history = history_list[-MAX_HISTORY:]

        chat_history_str = ""
        for msg in recent_history:
            if isinstance(msg, dict):
                role    = msg.get("role", "")
                content = str(msg.get("content", ""))
                if role == "user" and content:
                    chat_history_str += f"Patient: {content}\n"
                elif role == "assistant" and content:
                    chat_history_str += f"Mentallico: {content}\n"
            elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                # fallback للـ format القديم
                user_msg = str(msg[0]) if msg[0] else ""
                bot_msg  = str(msg[1]) if msg[1] else ""
                if user_msg and bot_msg:
                    chat_history_str += f"Patient: {user_msg}\nMentallico: {bot_msg}\n"

        full_response = ""
        for chunk in rag_chain.stream({
            "input":        user_query,
            "chat_history": chat_history_str,
        }):
            full_response += chunk
            yield full_response

    except Exception as e:
        yield f"An error occurred while connecting to the model. Error: {str(e)}"


# 11. Process New PDF
def process_new_pdf(file_path):
    global retriever, rag_chain

    try:
        if file_path is None:
            return "Please select a file first."

        loader = PyPDFLoader(file_path)
        docs   = loader.load()

        if not docs:
            return "The PDF appears to be empty or unreadable."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        vector_db.add_documents(chunks)

        retriever = vector_db.as_retriever(search_kwargs={"k": 6})
        rag_chain = _build_rag_chain(ret=retriever)

        return (
            f" Successfully processed '{os.path.basename(file_path)}'.\n"
            f" {len(docs)} pages → {len(chunks)} chunks added to the knowledge base.\n"
            f" Knowledge base updated and ready!"
        )

    except Exception as e:
        return f" Error processing file: {str(e)}"
