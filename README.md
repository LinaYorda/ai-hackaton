# Lawbotica: Your Legal Assistant Bot

**Lawbotica** is an AI-powered legal assistant designed to analyze legal documents, answer questions, and provide insightful responses. Built on state-of-the-art machine learning models, it combines natural language processing and vector search technology to make navigating legal documents seamless.

This project was developed as part of the AI + Digital Evidence Hackathon.It integrates resources like the [Leiden Guidelines](https://leiden-guidelines.com) to improve legal evidence analysis.

---

## Features

- **Q&A Dataset**  
  Preloaded with predefined questions and answers for quick reference.  
  Easy to extend by adding more questions and answers to the dataset.

- **PDF Document Analysis**  
  Processes legal documents in PDF format and extracts meaningful insights.  
  Supports various file types, including `.pdf`, `.txt`, `.docx`, and `.html`.

- **Falcon-7B Instruct**  
  Employs the Falcon-7B Instruct model for advanced natural language understanding.  
  Provides concise and context-aware answers to complex legal queries.

- **Vector Database with Chroma**  
  Leverages a vector search engine for similarity-based question answering.  
  Ensures fast and accurate retrieval of relevant document chunks.

---

## Getting Started

### Prerequisites

- Python 3.9 or above  
- GPU (optional but recommended for faster model performance)  
- Required Python libraries (listed in `requirements.txt`)


![Screenshot of Lawbotica](lawbotica.png)

### Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/lawbotica.git
   cd lawbotica
   pip install -r requirements.txt
   python lawbotica.py


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
├── requrements.txt # Requirements file
├── lawbotica.py # Main Python script
├── lawbotica.ipynb   # Main Jupyter Notebook
├── lawbotica_questions.xlsx    # Q&A dataset
├── README.md                   # Project documentation
├── legal_documents/            # Folder containing legal document PDFs
├── legal_chroma_db # Vector database
```

# Notes and Limitations

* This assistant is designed to help navigate legal documents and provide general insights.
It is not a substitute for professional legal advice.
* Responses are generated using an AI model trained on publicly available datasets.
While efforts have been made to ensure accuracy, results may vary.

# Contributing

* 