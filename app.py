import gradio as gr
from rag_engine import generate_answer, process_new_pdf, transcribe_audio


def handle_user_input(text, audio, history):
    text    = text    or ""
    history = history or []

    if audio:
        transcribed_text = transcribe_audio(audio)
        if isinstance(transcribed_text, str) and transcribed_text.startswith("Error"):
            return gr.update(), gr.update(value=None), history
        query = f"{text} {transcribed_text}".strip()
    else:
        query = text.strip()

    if not query:
        return gr.update(), gr.update(), history

    history.append({"role": "user", "content": query})
    return gr.update(value=""), gr.update(value=None), history


def generate_bot_response(history):
    history = history or []
    if not history:
        yield history
        return

    user_query = history[-1]["content"]

    history.append({"role": "assistant", "content": ""})

    for partial_response in generate_answer(user_query, history[:-1]):
        history[-1]["content"] = partial_response
        yield history


# ----------------- Build the Interface -----------------
with gr.Blocks(title="Mentallico AI") as app:

    gr.Markdown("""
        <center>
            <h1> Mentallico AI</h1>
            <p style="color: #555;">Your Advanced Psychiatric Diagnostic Assistant</p>
        </center>
    """)

    with gr.Row():

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=500,
                label="Mentallico Clinic",
                value=[]
            )

            with gr.Row():
                txt_input = gr.Textbox(
                    scale=7,
                    show_label=False,
                    placeholder="Type your message here...",
                    container=False
                )
                audio_input = gr.Audio(
                    scale=2,
                    sources=["microphone"],
                    type="filepath",
                    show_label=False,
                    container=False
                )
                submit_btn = gr.Button(
                    "Send 🚀",
                    scale=1,
                    variant="primary"
                )

            gr.Markdown(
                "⚠️ **Medical Disclaimer:** *Mentallico is an AI assistant designed for educational "
                "and preliminary diagnostic support based on uploaded literature. It does not replace "
                "professional medical advice, diagnosis, or treatment. In case of a medical or psychological "
                "emergency, please contact your local healthcare provider immediately.*"
            )

        with gr.Column(scale=1):
            gr.Markdown("###  Upload Medical Records")
            gr.Markdown("Upload a new patient record or medical paper for Mentallico to analyze and learn from instantly.")

            file_input    = gr.File(label="Select PDF File", file_types=[".pdf"])
            upload_button = gr.Button("Analyze Document ", variant="secondary")
            upload_status = gr.Textbox(label="Status", interactive=False, lines=4)

            upload_button.click(
                fn=process_new_pdf,
                inputs=file_input,
                outputs=upload_status
            )

            gr.Markdown("---")
            gr.Markdown(
                "###  Tips\n"
                "- Ask about symptoms, diagnoses, or treatments\n"
                "- You can speak  in English or Arabic\n"
                "- Upload new PDFs anytime to expand knowledge"
            )

    submit_btn.click(
        fn=handle_user_input,
        inputs=[txt_input, audio_input, chatbot],
        outputs=[txt_input, audio_input, chatbot],
        queue=True
    ).then(
        fn=generate_bot_response,
        inputs=chatbot,
        outputs=chatbot
    )

    txt_input.submit(
        fn=handle_user_input,
        inputs=[txt_input, audio_input, chatbot],
        outputs=[txt_input, audio_input, chatbot],
        queue=True
    ).then(
        fn=generate_bot_response,
        inputs=chatbot,
        outputs=chatbot
    )


if __name__ == "__main__":
    app.queue()
    app.launch(theme="soft")
