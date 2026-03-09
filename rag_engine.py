import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load local free Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Connect to the ChromaDB vector database
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Set up the Language Model (LLM) using Groq
os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0
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
    ("human", "{input}"),
])

# 5. Create the Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Generate Answer Function 
def generate_answer(user_query, return_raw_rag=False):
    try:
        # لو الواجهة باعتة True، هيرجع النص الخام
        if return_raw_rag:
            docs = retriever.invoke(user_query)
            
            if not docs:
                return "I could not find any information related to your question in the database."
            
            raw_response = " **Here is the extracted information directly from the documents:**\n\n"
            for i, doc in enumerate(docs):
                raw_response += f"**[Excerpt {i+1}]:**\n{doc.page_content}\n\n---\n"
            
            return raw_response
            
        else:
            # هنا بقى الرد الاحترافي بتاع الدكتور Mentallico
            response = rag_chain.invoke({"input": user_query})
            return response["answer"]
            
    except Exception as e:
        return f"An error occurred while connecting to the model. Error: {str(e)}"

# 7. Process New PDF Function (تم فصلها وضبط المسافات)
def process_new_pdf(file_path):
    try:
        if file_path is None:
            return "Please select a file first."
            
        # 1. قراءة الملف
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # 2. تقطيع الملف (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        # 3. إضافة القطع لقاعدة البيانات الحالية
        vector_db.add_documents(chunks)
        
        return f"Successfully processed and learned from the new PDF ({len(chunks)} chunks added)."
    except Exception as e:
        return f"Error processing file: {str(e)}"
