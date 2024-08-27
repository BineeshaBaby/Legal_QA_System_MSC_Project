# LEGA: Legal Question Answering System
A comprehensive system designed to automate and enhance legal research by utilizing Large Language Models (LLMs) and Natural Language Processing (NLP) techniques.

## Table of Contents
1. [Project Overview]()
2. [File Descriptions]()
3. [Input Data Format]()
4. [Output Data/Plots/Tables]()
5. [Parameters and Hyperparameters]()
6. [Relevant Issues]()
7. [Installation]()
8. [Usage]()
9. [Contributing]()
10. [License]()
11. [Contact]

## Project Overview
LEGA is a Legal Question Answering System designed to automate the research process by interpreting legal queries and summarizing relevant legal documents. This system leverages the power of advanced LLMs like GPT-4 combined with NLP techniques to provide accurate and contextually relevant answers to legal questions.

## File Descriptions

### 1. `pdf_extractor.py`
- **Purpose**: Extracts text from PDF documents and preprocesses it for further analysis.
- **Key Functions**:
  - `extract_text_from_pdf()`: Extracts raw text from a PDF file.
  - And other functions used to Clean and preprocess the extracted text for model input.

### 2. `storage.py`
- **Purpose**: Manages data storage, including embeddings and other processed data, using ChromaDB.
- **Key Functions**:
  - `sstore_embeddings_in_chromadb()`: Stores embeddings generated from text data.
  - And other functions Retrieves stored embeddings for similarity searches.

### 3. `chat_model.py`
- **Purpose**: Implements the GPT-4 model for generating responses to legal questions.
- **Key Functions**:
  - `generate_response()`: Uses GPT-4 to generate a response to a given legal query.
  - `summarize_document()`: Summarizes long legal documents using GPT-4.

### 4. `backend.py`
- **Purpose**: Handles the backend API for the application using FastAPI.
- **Key Functions**:
  - `query_legal_documents()`: API endpoint for querying legal documents.

### 5. `front_end.py`
- **Purpose**: Implements the frontend interface using Gradio, allowing users to interact with the system.
- **Key Functions**:
  - `create_interface()`: Sets up the Gradio interface for user interaction.

### 6. `metrics.py`
- **Purpose**: Calculates performance metrics such as BLEU, ROUGE, and Latency.
- **Key Functions**:
  - `calculate_bleu_score()`: Calculates the BLEU score for model outputs.
  - `calculate_rouge_score()`: Calculates the ROUGE score for model outputs.

### 7. `analysis.py`
- **Purpose**: Performs analysis on the model's performance and comparison between different models.
- **Key Functions**:
  - `compare_models()`: Compares the performance of GPT-4 and other models.
  - `visualize_performance()`: Generates plots to visualize model performance.

### 8. `gpt_vs_t5.py`
- **Purpose**: A specific script to compare GPT-4 against T5 models in terms of accuracy and efficiency.
- **Key Functions**:
  - `comparison()`: Runs the comparison between GPT-4 and T5 models and generates a report.

## Input Data Format
- **PDF Documents**: Raw legal texts in PDF format.
- **Text Files**: Preprocessed text data for embedding and analysis.
- **Query Input**: User-provided legal questions in plain text.

## Output Data/Plots/Tables
- **Summaries**: Text summaries of legal documents.
- **Model Responses**: Generated answers to legal questions.
- **Performance Metrics**: BLEU, ROUGE scores, and Latency values.
- **Comparison Graphs**: Visual comparisons between GPT-4 and GPT-3.5-turbo and T5 models.

## Parameters and Hyperparameters
- **GPT-4**: 
  - `temperature`: Controls the randomness of model outputs.
  - `max_tokens`: The maximum number of tokens to generate.
- **T5**:
  - `num_beams`: The number of beams for beam search.
  - `length_penalty`: Adjusts the length of generated sequences.
- **General**:
  - `batch_size`: Number of samples processed in one batch.
  - `learning_rate`: The step size for model weight updates.

## Relevant Issues
- **Memory Consumption**: GPT-4 models can be memory-intensive. Ensure adequate resources.
- **Latency**: Response times may vary depending on the complexity of the query.
- **Data Sensitivity**: Handle legal documents with care to maintain confidentiality and comply with privacy laws.

## Installation
To set up this project locally:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/BineeshaBaby/Legal_QA_System_MSC_Project.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd main
    ```

3. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv\Scripts\activate
    ```

4. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Set up environment variables**:
    -

6. **Run the application**:
    ```bash
    uvicorn backend:backend --reload
    ```

## Usage
1. **Access the application**:
   - Open your browser and go to `http://localhost:8000`.

2. **Interact with the system**:
   - Use the interface to input queries and receive answers.


## Contact
For questions or support:
- **Name**: Bineesha Baby


