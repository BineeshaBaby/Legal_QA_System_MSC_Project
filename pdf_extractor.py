<<<<<<< HEAD
import os
import json
import openai
import logging
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import concurrent.futures
import spacy

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {e}")
        return None
    return text

def create_overlapping_chunks(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def enhance_text_with_openai(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ],
            max_tokens=1500,
            temperature=0.7,
            n=1
        )
        generated_text = response.choices[0]['message']['content'].strip() if response.choices else None
        return generated_text
    except Exception as e:
        logging.error(f"Error enhancing text with OpenAI: {e}")
        return None

def enhance_texts_parallel(chunks):
    enhanced_texts = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(enhance_text_with_openai, chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                enhanced_texts.append(result)
            else:
                enhanced_texts.append("")
    return enhanced_texts

def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def create_metadata_for_chunks(document_name, text_chunks):
    try:
        metadata_list = []
        for i, chunk in enumerate(text_chunks):
            entities = perform_ner(chunk)
            metadata_list.append({"document_id": document_name, "chunk_id": i, "chunk_length": len(chunk), "entities": entities})
        return metadata_list
    except Exception as e:
        log_error(f"Error creating metadata for {document_name}: {str(e)}")
        return None

def save_combined_text_to_file(original_chunks, enhanced_chunks, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Original and Enhanced Text Chunks:\n\n")
            for original, enhanced in zip(original_chunks, enhanced_chunks):
                f.write("Original:\n%s\n" % original)
                f.write("Enhanced:\n%s\n\n" % enhanced)
        logging.info(f"Combined text chunks successfully saved to {file_path}")
    except Exception as e:
        log_error(f"Error saving text chunks to {file_path}: {str(e)}")

def save_metadata_to_file(metadata_list, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=4)
        logging.info(f"Metadata successfully saved to {file_path}")
    except Exception as e:
        log_error(f"Error saving metadata to {file_path}: {str(e)}")

def save_combined_data_to_json(original_chunks, enhanced_chunks, file_path):
    try:
        combined_data = [{"original": orig, "enhanced": enh} for orig, enh in zip(original_chunks, enhanced_chunks)]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Combined data successfully saved to {file_path}")
    except Exception as e:
        log_error(f"Error saving combined data to {file_path}: {str(e)}")

def log_error(message):
    logging.error(message)
    with open("error_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

if __name__ == "__main__":
    pdf_files = [
        r"C:\Users\BINEESHA BABY\Desktop\MSC Project\Legal_documents\Martial Law in England.pdf",
        r"C:\Users\BINEESHA BABY\Desktop\MSC Project\Legal_documents\Intellectual Property Law.pdf",
        r"C:\Users\BINEESHA BABY\Desktop\MSC Project\Legal_documents\Environmental Law.pdf"
    ]

    all_original_chunks = []
    all_enhanced_chunks = []
    all_metadata = []

    for PDF_FILE in pdf_files:
        document_name = os.path.basename(PDF_FILE).replace(".pdf", "")
        extracted_text = extract_text_from_pdf(PDF_FILE)
        
        if extracted_text:
            text_chunks = create_overlapping_chunks(extracted_text)
            enhanced_texts = enhance_texts_parallel(text_chunks)
            
            all_original_chunks.extend(text_chunks)
            all_enhanced_chunks.extend(enhanced_texts)
            metadata_list = create_metadata_for_chunks(document_name, text_chunks)
            if metadata_list:
                all_metadata.extend(metadata_list)
            
            logging.info(f"Processed {document_name}")
        else:
            logging.error(f"No text extracted from {PDF_FILE}")

    # Save combined text to a single file
    combined_text_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_text_chunks_all.txt"
    save_combined_text_to_file(all_original_chunks, all_enhanced_chunks, combined_text_file_path)

    # Save combined metadata to a single file
    combined_metadata_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_metadata_all.json"
    save_metadata_to_file(all_metadata, combined_metadata_file_path)

    # Save combined chunks to a single JSON file
    combined_json_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_chunks_all.json"
    save_combined_data_to_json(all_original_chunks, all_enhanced_chunks, combined_json_file_path)
=======
import os
import json
import openai
import logging
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import concurrent.futures
import spacy

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {e}")
        return None
    return text

def create_overlapping_chunks(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def enhance_text_with_openai(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ],
            max_tokens=1500,
            temperature=0.7,
            n=1
        )
        generated_text = response.choices[0]['message']['content'].strip() if response.choices else None
        return generated_text
    except Exception as e:
        logging.error(f"Error enhancing text with OpenAI: {e}")
        return None

def enhance_texts_parallel(chunks):
    enhanced_texts = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(enhance_text_with_openai, chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                enhanced_texts.append(result)
            else:
                enhanced_texts.append("")
    return enhanced_texts

def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def create_metadata_for_chunks(document_name, text_chunks):
    try:
        metadata_list = []
        for i, chunk in enumerate(text_chunks):
            entities = perform_ner(chunk)
            metadata_list.append({"document_id": document_name, "chunk_id": i, "chunk_length": len(chunk), "entities": entities})
        return metadata_list
    except Exception as e:
        log_error(f"Error creating metadata for {document_name}: {str(e)}")
        return None

def save_combined_text_to_file(original_chunks, enhanced_chunks, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Original and Enhanced Text Chunks:\n\n")
            for original, enhanced in zip(original_chunks, enhanced_chunks):
                f.write("Original:\n%s\n" % original)
                f.write("Enhanced:\n%s\n\n" % enhanced)
        logging.info(f"Combined text chunks successfully saved to {file_path}")
    except Exception as e:
        log_error(f"Error saving text chunks to {file_path}: {str(e)}")

def save_metadata_to_file(metadata_list, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=4)
        logging.info(f"Metadata successfully saved to {file_path}")
    except Exception as e:
        log_error(f"Error saving metadata to {file_path}: {str(e)}")

def save_combined_data_to_json(original_chunks, enhanced_chunks, file_path):
    try:
        combined_data = [{"original": orig, "enhanced": enh} for orig, enh in zip(original_chunks, enhanced_chunks)]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Combined data successfully saved to {file_path}")
    except Exception as e:
        log_error(f"Error saving combined data to {file_path}: {str(e)}")

def log_error(message):
    logging.error(message)
    with open("error_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

if __name__ == "__main__":
    pdf_files = [
        r"C:\Users\BINEESHA BABY\Desktop\MSC Project\Legal_documents\Martial Law in England.pdf",
        r"C:\Users\BINEESHA BABY\Desktop\MSC Project\Legal_documents\Intellectual Property Law.pdf",
        r"C:\Users\BINEESHA BABY\Desktop\MSC Project\Legal_documents\Environmental Law.pdf"
    ]

    all_original_chunks = []
    all_enhanced_chunks = []
    all_metadata = []

    for PDF_FILE in pdf_files:
        document_name = os.path.basename(PDF_FILE).replace(".pdf", "")
        extracted_text = extract_text_from_pdf(PDF_FILE)
        
        if extracted_text:
            text_chunks = create_overlapping_chunks(extracted_text)
            enhanced_texts = enhance_texts_parallel(text_chunks)
            
            all_original_chunks.extend(text_chunks)
            all_enhanced_chunks.extend(enhanced_texts)
            metadata_list = create_metadata_for_chunks(document_name, text_chunks)
            if metadata_list:
                all_metadata.extend(metadata_list)
            
            logging.info(f"Processed {document_name}")
        else:
            logging.error(f"No text extracted from {PDF_FILE}")

    # Save combined text to a single file
    combined_text_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_text_chunks_all.txt"
    save_combined_text_to_file(all_original_chunks, all_enhanced_chunks, combined_text_file_path)

    # Save combined metadata to a single file
    combined_metadata_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_metadata_all.json"
    save_metadata_to_file(all_metadata, combined_metadata_file_path)

    # Save combined chunks to a single JSON file
    combined_json_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_chunks_all.json"
    save_combined_data_to_json(all_original_chunks, all_enhanced_chunks, combined_json_file_path)
>>>>>>> b7bbe38 (adding files after end to end tesing)
