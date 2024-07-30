import os
import json
import logging
import openai
from model import load_text_chunks_and_metadata
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data_for_finetuning(text_chunks):
    """
    Prepare the data for fine-tuning-like functionality.
    
    Args:
        text_chunks: List of text chunks.
    
    Returns:
        List of prompt-completion pairs.
    """
    prompt_data = []
    for text in text_chunks:
        prompt_data.append({
            "prompt": text,
            "completion": " "  # Add appropriate completion text based on your data
        })
    return prompt_data

def upload_finetune_data(prompt_data):
    """
    Upload the fine-tune data to OpenAI.
    
    Args:
        prompt_data: List of dictionaries with 'prompt' and 'completion'.
    
    Returns:
        Uploaded file ID.
    """
    logging.info("Starting the data upload process...")
    try:
        # Create a file with the prompt data
        with open("fine_tune_data.jsonl", "w") as f:
            for entry in prompt_data:
                json.dump(entry, f)
                f.write("\n")

        # Upload the file to OpenAI
        file_response = openai.File.create(file=open("fine_tune_data.jsonl"), purpose='fine-tune')
        training_file_id = file_response['id']
        logging.info(f"Uploaded file with ID: {training_file_id}")
        return training_file_id
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI error: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    # Path to the text chunks and metadata files
    text_chunks_file = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_text_chunks.txt"
    metadata_list_file = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\metadata.json"

    # Load the text chunks and metadata from files
    logging.info("Starting the loading process...")
    text_chunks, metadata_list = load_text_chunks_and_metadata(text_chunks_file, metadata_list_file)

    # Prepare prompt data for fine-tuning-like functionality
    prompt_data = prepare_data_for_finetuning(text_chunks)

    # Upload the fine-tune data to OpenAI
    uploaded_file_id = upload_finetune_data(prompt_data)
    logging.info(f"Uploaded file ID: {uploaded_file_id}")
