import gradio as gr
from rag_engine import generate_answer, process_new_pdf

def chat_wrapper(message, history):
    return generate_answer(message)

with gr.Blocks(theme="soft", title="Mentallico AI") as app:
    gr.Markdown("<center><h1> Mentallico AI </h1></center>")
    gr.Markdown("<center>Your Advanced Psychiatric Diagnostic Assistant</center>")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.ChatInterface(
                fn=chat_wrapper,
                chatbot=gr.Chatbot(height=500)
            )
            
        with gr.Column(scale=1):
            gr.Markdown("###  Upload Medical Records")
            gr.Markdown("Upload a new patient record or medical paper for Mentallico to analyze.")
            
            file_input = gr.File(label="Select PDF File", file_types=[".pdf"])
            upload_button = gr.Button("Analyze Document", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False)
            
            upload_button.click(
                fn=process_new_pdf, 
                inputs=file_input, 
                outputs=upload_status
            )

if __name__ == "__main__":
    app.launch()
