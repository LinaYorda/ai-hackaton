import os
from markitdown import MarkItDown
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import torch
import logging
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Environment setup is complete and logging is configured.")

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load the Q&A data from Excel
file_path = "/content/lawbotica_questions.xlsx"
data = pd.read_excel(file_path)

# Clean and prepare the Q&A data
data.columns = ["Questions", "Answers", "Source"]  # Rename columns for consistency
questions = data["Questions"].dropna().tolist()
answers = data["Answers"].dropna().tolist()
sources = data["Source"].dropna().tolist()


print(f"Loaded {len(questions)} questions from the dataset!")

# Model name
model_name = "tiiuae/falcon-7b-instruct"


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)


tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)
model.eval()

print("✅ Falcon 7B Instruct loaded and ready!")


for i in range(min(3, len(questions))):
    print(f"Q{i+1}: {questions[i]}")
    print(f"A{i+1}: {answers[i]}")
    print(f"Source: {sources[i]}")
    print("-" * 40)

# Directory containing legal documents
DOCS_FOLDER = "/content/legal_documents"
DB_PATH = "/content/legal_chroma_db"


doc_files = [
    f for f in os.listdir(DOCS_FOLDER)
    if f.lower().endswith(('.pdf', '.docx', '.txt', '.html', '.pptx'))
]


def convert_and_split(file_path):
    md = MarkItDown()
    result = md.convert(file_path)
    text_content = result.text_content


    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text_content)
    return chunks

texts = []
metadatas = []


for doc_name in doc_files:
    file_path = os.path.join(DOCS_FOLDER, doc_name)
    logger.info(f"Processing {doc_name}...")
    try:

        chunks = convert_and_split(file_path)
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"source": doc_name})
    except Exception as e:
        logger.error(f"Error processing {doc_name}: {e}")

# Add Q&A data to the embeddings
for question, answer, source in zip(questions, answers, sources):
    texts.append(question)
    metadatas.append({"answer": answer, "source": source})

# Create embeddings and Chroma vectorstore
if texts:
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Combine document text and Q&A into the vectorstore
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=DB_PATH
    )
    vectorstore.persist()
    logger.info("✅ Chroma vectorstore created and persisted!")
else:
    logger.warning("No text found to embed. Please ensure your documents are text-based (not scanned images).")

def clean_redundancy(text):
    """
    Remove repeated phrases and sentences from the response.
    """

    sentences = text.split(". ")
    seen = set()
    cleaned = [sentence for sentence in sentences if sentence not in seen and not seen.add(sentence)]
    return ". ".join(cleaned).strip()

def answer_question(question, vectorstore, top_k=3, max_new_tokens=300, max_length=512):
    """
    Generate a concise, non-redundant, and complete answer using Falcon.
    """

    normalized_query = question.strip().lower()


    for i, original_question in enumerate(questions):
        if normalized_query == original_question.strip().lower():
            return f"{answers[i]} (Source: {sources[i]})"


    docs = vectorstore.similarity_search(query=normalized_query, k=top_k)
    if not docs:
        return "Sorry, I couldn't find relevant information for your question."


    unique_contexts = list({doc.page_content[:500] for doc in docs})
    if not unique_contexts:
        return "Sorry, no relevant information was found for your query."
    context_str = "\n\n".join(unique_contexts[:3])  # Use only the top 3 chunks for context

    # Build the prompt
    prompt = (
        "You are a helpful legal assistant.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        "Please summarize the relevant details concisely, without repeating ideas. Provide a complete and clear answer directly addressing the question.\n\n"
        "Answer:"
    )


    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.9,
        do_sample=True
    )

    # Decode and clean the answer
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:", 1)[1].strip()
    else:
        answer = generated_text.strip()

    return clean_redundancy(answer)


def gradio_interface(question):
    """
    Gradio wrapper for the answer_question function.
    """
    if not question.strip():
        return "Please enter a question to proceed."

    try:
        response = answer_question(question, vectorstore)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Footer content
footer_content = """
<div style="text-align: center; margin-top: 20px; color: red;">
    <h4>Disclaimer</h4>
    <p>
        This application uses Falcon-7B Instruct, a large language model, to generate responses. While LLMs are highly capable,
        they may produce inaccurate, irrelevant, or outdated information. Use outputs responsibly and verify any critical details independently.
    </p>
    <p>
        This chatbot was trained and developed as a part of the <strong>AI + Digital Evidence Hackathon</strong>.
    </p>
    <p>
        <strong>Model:</strong> Falcon-7B Instruct | <strong>Embeddings:</strong> Sentence-Transformers (all-MiniLM-L6-v2)
        <strong>Libraries:</strong> Transformers, LangChain, Chroma, Gradio
        <strong>Key Parameters:</strong> max_new_tokens=300, temperature=0.2, top_p=0.9
    </p>
</div>
"""

# Use Gradio Blocks for layout customization
with gr.Blocks(css=".gradio-container { font-family: Arial, sans-serif; }") as app:
    gr.Markdown(
        """
        <div style="text-align: center; color: blue;">
            <h1>Lawbotica: Your Legal Document Assistant</h1>
            <p>
                <br>This assistant was trained on legal documents, including those available at
                <a href="https://leiden-guidelines.com" target="_blank">Leiden Guidelines</a>.
            </p>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            question_input = gr.Textbox(
                label="Enter Your Legal Question",
                placeholder="Ask a question about your documents...",
                lines=10  # Increased size
            )
        with gr.Column(scale=1, min_width=400):
            answer_output = gr.Textbox(
                label="Answer",
                placeholder="The assistant's answer will appear here.",
                lines=10  # Increased size
            )
    gr.Button("Submit", elem_id="submit-btn").click(gradio_interface, inputs=question_input, outputs=answer_output)
    gr.Markdown(footer_content)

# Add custom CSS for blue styling
app.css = """
#submit-btn {
    background-color: blue;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    padding: 10px 15px;
}
"""

# Launch Gradio app
app.launch(share=True)