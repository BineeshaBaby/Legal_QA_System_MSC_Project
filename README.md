# Legal_QA_System_MSC_Project
A specialized system for answering legal questions using natural language processing and machine learning techniques 
## Overview
This project processes legal documents by extracting text from PDFs, enhancing the text using OpenAI, creating metadata, cleaning and tokenizing the text, and storing embeddings in ChromaDB. It also includes querying capabilities to search for relevant information.

## Setup
1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install the dependencies:
4. Create a `.env` file in the root directory and add your OpenAI API key:

## Usage
1. Run `extract_text.py` to extract text from a PDF and create overlapping chunks.
2. Run `enhance_text.py` to enhance the text chunks using OpenAI.
3. Run `metadata.py` to create and save metadata for the text chunks.
4. Run `clean_tokenize.py` to clean and tokenize the text chunks.
5. Run `model.py` to initialize the Sentence Transformer model, store embeddings in ChromaDB, and query ChromaDB.
6. Run `chat_model.py` to generate responses using the OpenAI API.

## Files
- `extract_text.py`: Extracts text from a PDF file and creates overlapping chunks of text.
- `enhance_text.py`: Enhances text chunks using the OpenAI API.
- `metadata.py`: Creates and saves metadata for text chunks.
- `clean_tokenize.py`: Cleans and tokenizes text chunks using spaCy and NLTK.
- `model.py`: Initializes the Sentence Transformer model, stores embeddings in ChromaDB, and queries ChromaDB.
- `chat_model.py`: Uses the OpenAI API to generate responses based on predefined chat messages.
- `.env`: Stores environment variables such as the OpenAI API key.
- `requirements.txt`: Lists all the dependencies required for the project.
- `combined_text_chunks.txt`: Contains the combined text chunks extracted from the PDF.
- `metadata.json`: Contains metadata for each text chunk.


