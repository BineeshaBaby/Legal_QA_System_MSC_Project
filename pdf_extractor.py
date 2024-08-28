import os
import json
import openai
import logging
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import concurrent.futures
import spacy

# Load environment variables from a .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Configure logging to log both info and error messages with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the SpaCy model for natural language processing (NLP) tasks
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF, or None if an error occurs.
    """
    text = ""  # Initialize an empty string to store the extracted text
    try:
        # Open the PDF file in binary mode
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            # Iterate through each page and extract text
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {e}")  # Log any errors that occur
        return None
    return text

def create_overlapping_chunks(text, chunk_size=1000, overlap=200):
    """
    Splits text into overlapping chunks.

    Args:
        text (str): The text to be split into chunks.
        chunk_size (int): The number of words per chunk.
        overlap (int): The number of overlapping words between chunks.

    Returns:
        list: A list of text chunks.
    """
    words = text.split()  # Split text into words
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        # Create a chunk of the specified size
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        # Break the loop if the end of the text is reached
        if i + chunk_size >= len(words):
            break
    return chunks

def enhance_text_with_openai(text):
    """
    Enhances text using OpenAI's GPT-4 model.

    Args:
        text (str): The text to be enhanced.

    Returns:
        str: The enhanced text, or None if an error occurs.
    """
    try:
        # Generate an enhanced version of the text using OpenAI's GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ],
            max_tokens=1500,  # Set the maximum number of tokens in the response
            temperature=0.7,  # Control the creativity of the output
            n=1  # Generate only one response
        )
        # Extract and return the generated text
        generated_text = response.choices[0]['message']['content'].strip() if response.choices else None
        return generated_text
    except Exception as e:
        logging.error(f"Error enhancing text with OpenAI: {e}")  # Log any errors that occur
        return None

def enhance_texts_parallel(chunks):
    """
    Enhances a list of text chunks in parallel using OpenAI's GPT-4 model.

    Args:
        chunks (list): List of text chunks to be enhanced.

    Returns:
        list: A list of enhanced text chunks.
    """
    enhanced_texts = []
    # Use a ThreadPoolExecutor to enhance text chunks in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(enhance_text_with_openai, chunk) for chunk in chunks]
        # Collect the results as they are completed
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                enhanced_texts.append(result)
            else:
                enhanced_texts.append("")  # Append an empty string if the enhancement fails
    return enhanced_texts

def perform_ner(text):
    """
    Performs Named Entity Recognition (NER) on the given text using SpaCy.

    Args:
        text (str): The text on which to perform NER.

    Returns:
        list: A list of tuples, each containing an entity and its label.
    """
    doc = nlp(text)  # Process the text with SpaCy's NLP model
    # Extract entities and their labels from the processed text
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def create_metadata_for_chunks(document_name, text_chunks):
    """
    Creates metadata for each chunk of text.

    Args:
        document_name (str): The name of the document.
        text_chunks (list): List of text chunks.

    Returns:
        list: A list of metadata dictionaries for each chunk.
    """
    try:
        metadata_list = []
        for i, chunk in enumerate(text_chunks):
            # Perform NER on each chunk to extract entities
            entities = perform_ner(chunk)
            # Create a metadata dictionary for each chunk
            metadata_list.append({
                "document_id": document_name,
                "chunk_id": i,
                "chunk_length": len(chunk),
                "entities": entities
            })
        return metadata_list
    except Exception as e:
        log_error(f"Error creating metadata for {document_name}: {str(e)}")  # Log any errors that occur
        return None

def save_combined_text_to_file(original_chunks, enhanced_chunks, file_path):
    """
    Saves original and enhanced text chunks to a file.

    Args:
        original_chunks (list): List of original text chunks.
        enhanced_chunks (list): List of enhanced text chunks.
        file_path (str): Path to the file where the combined text will be saved.
    """
    try:
        # Open the file in write mode with UTF-8 encoding
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Original and Enhanced Text Chunks:\n\n")
            for original, enhanced in zip(original_chunks, enhanced_chunks):
                f.write("Original:\n%s\n" % original)
                f.write("Enhanced:\n%s\n\n" % enhanced)
        logging.info(f"Combined text chunks successfully saved to {file_path}")
    except Exception as e:
        log_error(f"Error saving text chunks to {file_path}: {str(e)}")  # Log any errors that occur

def save_metadata_to_file(metadata_list, file_path):
    """
    Saves metadata to a file.

    Args:
        metadata_list (list): List of metadata dictionaries.
        file_path (str): Path to the file where metadata will be saved.
    """
    try:
        # Open the file in write mode with UTF-8 encoding
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=4)
        logging.info(f"Metadata successfully saved to {file_path}")
    except Exception as e:
        log_error(f"Error saving metadata to {file_path}: {str(e)}")  # Log any errors that occur

def save_combined_data_to_json(original_chunks, enhanced_chunks, file_path):
    """
    Saves combined original and enhanced text chunks to a JSON file.

    Args:
        original_chunks (list): List of original text chunks.
        enhanced_chunks (list): List of enhanced text chunks.
        file_path (str): Path to the JSON file where combined data will be saved.
    """
    try:
        # Combine original and enhanced chunks into a list of dictionaries
        combined_data = [{"original": orig, "enhanced": enh} for orig, enh in zip(original_chunks, enhanced_chunks)]
        # Save the combined data to a JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Combined data successfully saved to {file_path}")
    except Exception as e:
        log_error(f"Error saving combined data to {file_path}: {str(e)}")  # Log any errors that occur

def log_error(message):
    """
    Logs an error message to the console and to a file.

    Args:
        message (str): The error message to be logged.
    """
    logging.error(message)
    # Append the error message to the error log file
    with open("error_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

if __name__ == "__main__":
    # List of PDF files to process
    pdf_files = [
        r"C:\Users\BINEESHA BABY\Desktop\MSC Project\Legal_documents\Martial Law in England.pdf",
        r"C:\Users\BINEESHA BABY\Desktop\MSC Project\Legal_documents\Intellectual Property Law.pdf",
        r"C:\Users\BINEESHA BABY\Desktop\MSC Project\Legal_documents\Environmental Law.pdf"
    ]

    # Initialize lists to store all chunks and metadata
    all_original_chunks = []
    all_enhanced_chunks = []
    all_metadata = []

    # Process each PDF file
    for pdf_file in pdf_files:
        document_name = os.path.basename(pdf_file).replace(".pdf", "")  # Extract document name from file path
        extracted_text = extract_text_from_pdf(pdf_file)  # Extract text from the PDF
        
        if extracted_text:
            text_chunks = create_overlapping_chunks(extracted_text)  # Create overlapping chunks from extracted text
            enhanced_texts = enhance_texts_parallel(text_chunks)  # Enhance the chunks using OpenAI
            
            # Extend the lists with the results from this document
            all_original_chunks.extend(text_chunks)
            all_enhanced_chunks.extend(enhanced_texts)
            metadata_list = create_metadata_for_chunks(document_name, text_chunks)  # Generate metadata
            if metadata_list:
                all_metadata.extend(metadata_list)
            
            logging.info(f"Processed {document_name}")  # Log successful processing
        else:
            logging.error(f"No text extracted from {pdf_file}")  # Log an error if text extraction fails

    # Save combined text to a single file
    combined_text_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_text_chunks_all.txt"
    save_combined_text_to_file(all_original_chunks, all_enhanced_chunks, combined_text_file_path)

    # Save combined metadata to a single file
    combined_metadata_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_metadata_all.json"
    save_metadata_to_file(all_metadata, combined_metadata_file_path)

    # Save combined chunks to a single JSON file
    combined_json_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_chunks_all.json"
    save_combined_data_to_json(all_original_chunks, all_enhanced_chunks, combined_json_file_path)
