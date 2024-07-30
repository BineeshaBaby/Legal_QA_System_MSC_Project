import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_metadata_for_chunks(document_name, text_chunks):
    try:
        metadata_list = [{"document_id": document_name, "chunk_id": i, "chunk_length": len(chunk)} for i, chunk in enumerate(text_chunks)]
        return metadata_list
    except Exception as e:
        log_error(f"Error creating metadata for {document_name}: {str(e)}")
        return None

def save_metadata_to_file(metadata_list, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=4)
        logging.info(f"Metadata successfully saved to {file_path}")
    except Exception as e:
        log_error(f"Error saving metadata to {file_path}: {str(e)}")

def save_text_chunks_to_file(text_chunks, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for chunk in text_chunks:
                f.write("%s\n" % chunk)
        logging.info(f"Text chunks successfully saved to {file_path}")
    except Exception as e:
        log_error(f"Error saving text chunks to {file_path}: {str(e)}")

def load_text_chunks(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]
    except Exception as e:
        log_error(f"Error loading text chunks from {file_path}: {str(e)}")
        return []

def log_error(message):
    logging.error(message)
    with open("error_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

if __name__ == "__main__":
    document_name = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\Legal_documents\Martial Law in England.pdf"
    text_chunks_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_text_chunks.txt"
    
    text_chunks = load_text_chunks(text_chunks_file_path)

    if text_chunks:
        metadata_list = create_metadata_for_chunks(document_name, text_chunks)
        
        if metadata_list:
            logging.info("Metadata Created:")
            metadata_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\metadata.json"
            save_metadata_to_file(metadata_list, metadata_file_path)

            processed_chunks_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\processed_chunks.txt"
            save_text_chunks_to_file(text_chunks, processed_chunks_file_path)
        else:
            logging.error("Failed to create metadata.")
    else:
        logging.error("No text chunks loaded.")

