import os
import pandas as pd
import fitz  # PyMuPDF

def load_data(train_path, valid_path):
    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)
    return train_data['text'].tolist(), train_data['label'].tolist(), valid_data['text'].tolist(), valid_data['label'].tolist()

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def get_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(file_path)
            documents.append(text)
    return documents

def load_legal_documents(directory):
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as file:
                documents[filename] = file.read()
    return documents