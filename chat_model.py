import os 
import openai 
from dotenv import load_dotenv  
import logging 
import json  

# Load environment variables
load_dotenv()  # Loads environment variables from a .env file into the system's environment variables.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Retrieves the OpenAI API key from the environment variables.

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY  # Sets the OpenAI API key for use in API calls.

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Configures logging to display messages at the INFO level or higher.
# The format specifies how the log messages will be displayed, including the timestamp, the log level, and the message.

def load_knowledge_base(file_path):
    """
    Load the knowledge base from a specified file.

    Args:
        file_path (str): Path to the JSON file containing the knowledge base.

    Returns:
        dict: The loaded knowledge base as a dictionary, or an empty dictionary if an error occurs.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:  # Opens the specified file in read mode with UTF-8 encoding.
            return json.load(f)  # Loads the content of the file as a JSON object and returns it.
    except FileNotFoundError:  # Handles the case where the file does not exist.
        logging.error(f"File not found: {file_path}")  # Logs an error message indicating that the file was not found.
        return {}  # Returns an empty dictionary if the file is not found.
    except json.JSONDecodeError:  # Handles the case where the file content cannot be decoded as JSON.
        logging.error(f"Error decoding JSON from the file: {file_path}")  # Logs an error message indicating a JSON decoding issue.
        return {}  # Returns an empty dictionary if JSON decoding fails.
    except Exception as e:  # Handles any other exceptions that may occur.
        logging.error(f"Error loading knowledge base: {e}")  # Logs a generic error message.
        return {}  # Returns an empty dictionary if any other error occurs.

def summarize_text(text, max_tokens=3000):
    """
    Summarizes a given text using OpenAI's GPT-4 model.

    Args:
        text (str): The text to summarize.
        max_tokens (int): The maximum number of tokens (words/pieces of words) for the summary.
                         - This limits the length of the generated text.
                         - A typical token is about 4 characters of English text, or 3/4 of a word.
                         - Example: The word "Hello" counts as 1 token, but "ChatGPT is great!" counts as 5 tokens.
                         - Setting max_tokens too low might truncate the output, while setting it high allows for more detailed responses.
        temperature (float): Controls the creativity of the output.
                             - Lower values (close to 0) make the output more deterministic and focused.
                             - Higher values (closer to 1) make the output more random and creative.
                             - Example: A temperature of 0.7 will provide some variety, while a temperature of 0.1 will result in more predictable, conservative responses.

    Returns:
        str: The summarized text, or None if an error occurs.
    """
    try:
        response = openai.ChatCompletion.create(  # Calls the OpenAI API to generate a text summary.
            model="gpt-4",  # Specifies that the GPT-4 model should be used.
            messages=[  # Defines the conversation history that the model will see.
                {"role": "system", "content": "You are a helpful assistant. Please summarize the following text."},
                {"role": "user", "content": text}
            ],
            max_tokens=max_tokens,  # Limits the number of tokens (output length) the model can generate.
            temperature=0.5,  # Sets the creativity level to a moderate value.
        )
        summary = response.choices[0]['message']['content'].strip() if response.choices else None
        # Extracts and strips the generated summary from the API response.
        return summary  # Returns the summary.
    except openai.error.OpenAIError as e:  # Handles any errors returned by the OpenAI API.
        logging.error(f"Error summarizing text with OpenAI: {e}")  # Logs an error message.
        return None  # Returns None if an error occurs.

def generate_response_gpt4(context, question):
    """
    Generate a response using GPT-4 based on the provided context and question.
    
    Args:
        context (str): The context to be used for generating the response.
        question (str): The user's question.
    
    Returns:
        str: The generated response from the assistant.
    """
    # Construct the prompt with the context and question
    full_prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer strictly using the provided context:"
    # Combines the context and question into a single string prompt.

    # Create the conversation history for the GPT-4 model
    chat_messages = [
        {"role": "system", "content": "You are a legal assistant. Only use the provided context to answer the questions."},
        {"role": "user", "content": full_prompt}
    ]
    # Prepares the conversation history for the model, instructing it to use only the provided context to answer the question.

    try:
        # Generate a response from GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Specifies the use of the GPT-4 model.
            messages=chat_messages,  # Sends the conversation history to the model.
            max_tokens=1500,  # Restricts the response length (typically long enough for detailed answers).
            temperature=0.7  # Slightly more creative responses but still grounded.
        )
        return response.choices[0]['message']['content']  # Extracts and returns the generated response from the API.
    except openai.error.OpenAIError as e:  # Handles any errors returned by the OpenAI API.
        logging.error(f"An error occurred: {e}")  # Logs an error message.
        return "An error occurred while generating the response."  # Returns an error message to the user.

def save_summarized_context(summarized_context, text_file_path, json_file_path):
    """
    Save the summarized context to both a text file and a JSON file.

    Args:
        summarized_context (str): The summarized text to save.
        text_file_path (str): Path to the text file for saving the summarized context.
        json_file_path (str): Path to the JSON file for saving the summarized context.
    """
    try:
        # Save the summarized context to a text file
        with open(text_file_path, "w", encoding="utf-8") as text_file:  # Opens the text file in write mode.
            text_file.write(summarized_context)  # Writes the summarized context to the text file.
        logging.info(f"Summarized context successfully saved to {text_file_path}")  # Logs a success message.

        # Save the summarized context to a JSON file
        with open(json_file_path, "w", encoding="utf-8") as json_file:  # Opens the JSON file in write mode.
            json.dump({"summarized_context": summarized_context}, json_file, ensure_ascii=False, indent=4)
            # Dumps the summarized context as JSON into the file.
        logging.info(f"Summarized context successfully saved to {json_file_path}")  # Logs a success message.
    except Exception as e:  # Handles any exceptions that may occur during file operations.
        logging.error(f"Error saving summarized context: {e}")  # Logs an error message if something goes wrong.

def main():
    """
    Main function to load the knowledge base, summarize the context if needed,
    and interactively generate responses to legal questions using GPT-4.
    """
    knowledge_base_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_chunks_all.json"
    # Path to the knowledge base file.

    knowledge_base = load_knowledge_base(knowledge_base_file_path)  # Loads the knowledge base from the specified file.

    if not knowledge_base:  # Checks if the knowledge base was loaded successfully.
        logging.error("Knowledge base is empty or could not be loaded.")  # Logs an error if the knowledge base is empty.
        return  # Exits the function if there was an issue loading the knowledge base.

    # Combine all contexts into a single string
    context = " ".join([item.get("enhanced", "") for item in knowledge_base])
    # Combines all the "enhanced" contexts from the knowledge base into a single string.

    # Check if the combined context is too long and needs to be summarized in parts
    if len(context.split()) > 8000:  # Checks if the context is longer than 8000 words.
        logging.info("Context is too long, summarizing in smaller parts...")
        # Logs that the context will be summarized in smaller parts.

        # Split the context into smaller parts to avoid exceeding token limits
        parts = [context[i:i + 9000] for i in range(0, len(context), 9000)]
        # Splits the context into parts, each of which is 9000 characters long.

        summarized_parts = []  # Initializes an empty list to store the summarized parts.

        for i, part in enumerate(parts):  # Iterates over each part of the context.
            logging.info(f"Summarizing part {i + 1}/{len(parts)}...")  # Logs the progress of summarization.
            summary = summarize_text(part, max_tokens=5000)  # Summarizes each part with a maximum of 5000 tokens.
            if summary:  # Checks if the summarization was successful.
                summarized_parts.append(summary)  # Adds the summary to the list of summarized parts.
            else:
                logging.error(f"Failed to summarize part {i + 1}.")  # Logs an error if summarization failed.

        # Combine all summarized parts
        context = " ".join(summarized_parts)  # Combines all the summarized parts into a single string.
        if not context:  # Checks if the context is still empty after summarization.
            logging.error("Failed to summarize context.")  # Logs an error if summarization failed completely.
            return  # Exits the function if summarization was unsuccessful.

        # Save summarized context to files
        summarized_context_text_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\summarized_context.txt"
        summarized_context_json_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\summarized_context.json"
        save_summarized_context(context, summarized_context_text_file_path, summarized_context_json_file_path)
    

    # Interactively generate responses to user questions
    while True:  # Starts an infinite loop to interact with the user.
        question = input("Enter your legal question (or type 'exit' to quit): ")  # Prompts the user for a question.
        if question.lower() == 'exit':  # Checks if the user wants to exit the loop.
            break  # Exits the loop if the user types 'exit'.

        response = generate_response_gpt4(context, question)  # Generates a response to the user's question using GPT-4.
        logging.info(f"Generated response: {response}")  # Logs the generated response.
        print(response)  # Prints the response to the user.

if __name__ == "__main__":
    main()  # Calls the main function to start the program.

