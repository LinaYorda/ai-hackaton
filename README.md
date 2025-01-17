# Lawbotica: Legal Assistant Bot

This project is a **Jupyter Notebook-based legal assistant** that uses machine learning models to analyze legal documents and answer questions from a dataset or PDFs.

## Features
- **Q&A Dataset**: Includes predefined questions and answers for quick reference.
- **PDF Analysis**: Processes legal document PDFs and retrieves relevant context.
- **Falcon-7B Model**: Utilizes Falcon-7B Instruct for natural language processing and response generation.
- **Vector Database**: Employs Chroma for efficient similarity search.

---

## How to Use
1. **Open the Notebook**:
   Download or clone this repository, then open `legal_assistant_bot.ipynb` in Jupyter Notebook or JupyterLab.

2. **Run the Notebook**:
   Execute the notebook cells in order:
   - **Step 1**: Load and process the Q&A dataset.
   - **Step 2**: Process legal document PDFs into a vector database.
   - **Step 3**: Ask and answer questions using the Falcon-7B model.

---

## File Structure
```plaintext
legal-assistant-bot/
├── legal_assistant_bot.ipynb   # Main Jupyter Notebook
├── lawbotica_questions.xlsx    # Q&A dataset
├── README.md                   # Project documentation
├── legal_documents/            # Folder containing legal document PDFs
