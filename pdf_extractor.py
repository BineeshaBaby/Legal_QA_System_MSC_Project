import os
import openai
import logging
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def enhance_text_with_openai(text, api_key):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
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

def save_texts_to_file(original_chunks, enhanced_chunks, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Original Chunks:\n")
            for chunk in original_chunks:
                f.write("%s\n" % chunk)
            f.write("\nEnhanced Chunks:\n")
            for chunk in enhanced_chunks:
                f.write("%s\n" % chunk)
        logging.info(f"Text chunks and enhanced text successfully saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving text chunks and enhanced text to {file_path}: {str(e)}")

if __name__ == "__main__":
    PDF_FILE = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\Legal_documents\Martial Law in England.pdf"
    
    extracted_text = extract_text_from_pdf(PDF_FILE)
    
    if extracted_text:
        logging.info("Extracted Text:\n")
        logging.info(extracted_text)

        text_chunks = create_overlapping_chunks(extracted_text)
        
        enhanced_texts = []
        for chunk in text_chunks:
            enhanced_text = enhance_text_with_openai(chunk, OPENAI_API_KEY)
            if enhanced_text:
                enhanced_texts.append(enhanced_text)
            else:
                logging.warning("Failed to enhance text with OpenAI.")
        
        combined_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_text_chunks.txt"
        save_texts_to_file(text_chunks, enhanced_texts, combined_file_path)
        
        logging.info("Original and Enhanced Texts have been saved to the file.")
    else:
        logging.error(f"No text extracted from {PDF_FILE}")

