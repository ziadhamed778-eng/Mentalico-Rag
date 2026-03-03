import gradio as gr
from rag_engine import generate_answer

# Wrapper function to match Gradio's ChatInterface system
def chat_wrapper(message, history):
    # message: User's question
    # history: Chat history 
    # لغينا خيار الـ Raw خالص عشان نجبر الكود دايماً يستخدم دكتور Mentallico
    return generate_answer(message)

# Interface design and branding for the graduation project
app = gr.ChatInterface(
    fn=chat_wrapper,
    title="Mentallico AI ",
    description="Welcome to Mentallico. I am your AI psychiatrist. Ask me any medical question, and I will analyze the database to provide a professional, empathetic answer.")

if __name__ == "__main__":
    app.launch()