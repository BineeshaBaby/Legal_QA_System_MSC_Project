import os
import openai
from dotenv import load_dotenv
import logging
import json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the knowledge base
def load_knowledge_base(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error loading knowledge base: {e}")
        return []

def summarize_text(text, max_tokens=2000):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please summarize the following text."},
                {"role": "user", "content": text}
            ],
            max_tokens=max_tokens,
            temperature=0.5,
        )
        summary = response.choices[0]['message']['content'].strip() if response.choices else None
        return summary
    except openai.error.OpenAIError as e:
        logging.error(f"Error summarizing text with OpenAI: {e}")
        return None

def generate_response(context, question):
    """
    Generate a response using GPT-4 based on the provided context and question.
    
    Args:
        context (str): The context to be used for generating the response.
        question (str): The user's question.
    
    Returns:
        str: The generated response from the assistant.
    """
    full_prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer strictly using the provided context:"

    chat_messages = [
        {"role": "system", "content": "You are a legal assistant. Only use the provided context to answer the questions."},
        {"role": "user", "content": full_prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=chat_messages,
            max_tokens=1500,
            temperature=0.7
        )
        return response.choices[0]['message']['content']
    except openai.error.OpenAIError as e:
        logging.error(f"An error occurred: {e}")
        return "An error occurred while generating the response."

def save_summarized_context(summarized_context, text_file_path, json_file_path):
    try:
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(summarized_context)
        logging.info(f"Summarized context successfully saved to {text_file_path}")

        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump({"summarized_context": summarized_context}, json_file, ensure_ascii=False, indent=4)
        logging.info(f"Summarized context successfully saved to {json_file_path}")
    except Exception as e:
        logging.error(f"Error saving summarized context: {e}")

def main():
    knowledge_base_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_chunks_all.json"
    knowledge_base = load_knowledge_base(knowledge_base_file_path)

    if not knowledge_base:
        logging.error("Knowledge base is empty or could not be loaded.")
        return

    # Combine all contexts into a single string and summarize if necessary
    context = " ".join([item.get("enhanced", "") for item in knowledge_base])

    # Summarize context in parts if it is too long
    if len(context.split()) > 8000:
        logging.info("Context is too long, summarizing in parts...")
        parts = [context[i:i+4000] for i in range(0, len(context), 4000)]
        summarized_parts = [summarize_text(part, max_tokens=2000) for part in parts]
        context = " ".join(summarized_parts)
        if not context:
            logging.error("Failed to summarize context.")
            return
        
        # Save summarized context to files
        summarized_context_text_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\summarized_context.txt"
        summarized_context_json_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\summarized_context.json"
        save_summarized_context(context, summarized_context_text_file_path, summarized_context_json_file_path)

    while True:
        question = input("Enter your legal question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        response = generate_response(context, question)
        logging.info(f"Generated response: {response}")
        print(response)

if __name__ == "__main__":
    main()
