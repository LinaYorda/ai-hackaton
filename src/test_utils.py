from chatbot.utils import get_documents_from_folder

# Path to the folder containing your PDF documents
folder_path = 'data/legal_documents'

# Extract text from all PDF documents in the folder
documents = get_documents_from_folder(folder_path)

# Print the extracted text
for i, doc in enumerate(documents):
    print(f"Document {i+1}:\n{doc[:500]}...\n")  # Print the first 500 characters of each document