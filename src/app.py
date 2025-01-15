import os
from chatbot.model import LegalBertChatbot
from chatbot.utils import get_documents_from_folder
import gradio as gr

def ask_question(query):
    context = " ".join(documents)
    response = chatbot.generate_response(query, context)
    return response

if __name__ == "__main__":
    # Initialize chatbot
    chatbot = LegalBertChatbot()
    
    # Load documents
    documents = get_documents_from_folder('data/legal_documents')
    
    # Create Gradio interface
    demo = gr.Interface(
        fn=ask_question,
        inputs=[
            gr.Textbox(
                label="Your Legal Question",
                placeholder="Ask any question about your legal documents...",
                lines=3
            )
        ],
        outputs=[
            gr.Textbox(
                label="Answer",
                lines=10
            )
        ],
        title="Legal Document Assistant",
        description="This AI assistant can answer questions about your legal documents."
    )
    
    # Launch interface
    demo.launch(share=True)