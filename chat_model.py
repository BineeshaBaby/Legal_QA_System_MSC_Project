import os
import openai
from dotenv import load_dotenv
import logging
from sklearn.model_selection import train_test_split

# Load environment variables from a .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_response(chat_messages):
    """
    Generate a response using GPT-4 based on the provided chat messages.
    
    Args:
        chat_messages (list): List of chat messages in the format required by OpenAI API.
    
    Returns:
        str: The generated response from the assistant.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=chat_messages
        )
        return response.choices[0]['message']['content']
    except openai.error.OpenAIError as e:
        logging.error(f"An error occurred: {e}")
        return "An error occurred while generating the response."

def validate_model(validation_data):
    """
    Validate the model using the validation dataset.
    
    Args:
        validation_data (list): List of validation data in the format required by OpenAI API.
    
    Returns:
        list: List of validation results.
    """
    results = []
    for chat_messages in validation_data:
        response = generate_response(chat_messages)
        results.append(response)
    return results

def main():
    # Summarized text from your documents
    summarized_text_1 = (
        "Martial law in England originates from the Court of the Constable and Marshal. "
        "It had two main branches of jurisdiction: appeals of death or murder committed beyond the seas, "
        "and offences of soldiers. Its importance declined during the Tudor period due to rivalry with common law courts "
        "and Parliament's disfavor."
    )

    summarized_text_2 = (
        "The Magna Carta, signed in 1215, is a foundational document in English law. "
        "It established the principle that everyone, including the king, was subject to the law. "
        "It guaranteed the rights of individuals and the right to justice and a fair trial."
    )

    # Define multiple chat messages for training and validation
    chat_messages_1 = [
        {"role": "system", "content": "You are a legal assistant."},
        {"role": "user", "content": "What is martial law, and where does it originate from in England?"},
        {"role": "assistant", "content": "Martial law in England originates from the law administered by the Court of the Constable and Marshal, described as the fountain of Marshal Law."},
        {"role": "user", "content": "What were the two main branches of jurisdiction for the Court of the Constable and Marshal?"},
        {"role": "assistant", "content": "The two main branches were: (1) Jurisdiction over appeals of death or murder committed beyond the seas, and (2) the offences and miscarriages of soldiers contrary to the laws and rules of the army."},
        {"role": "user", "content": "Why did the importance and use of the Court of Constable and Marshal decline during the Tudor period?"},
        {"role": "assistant", "content": "The decline was due to the common law courts viewing it as a rival to their authority, and Parliament's disfavor, culminating in its downfall in 1640."},
        {"role": "user", "content": summarized_text_1}
    ]

    chat_messages_2 = [
        {"role": "system", "content": "You are a legal assistant."},
        {"role": "user", "content": "How was rebellion handled under martial law in England?"},
        {"role": "assistant", "content": "During the rebellion of Wat Tyler in the reign of Richard II, many were executed without trial. An Act of Indemnity was passed to exempt the king's supporters from responsibility for excesses committed in suppressing the rebellion."},
        {"role": "user", "content": "Was there a legal framework to justify such actions?"},
        {"role": "assistant", "content": "While martial law allowed for such actions in times of rebellion or insurrection, it was considered illegal to apply it once order had been restored. Executions carried out under martial law after the fact were deemed unlawful."}
    ]

    chat_messages_3 = [
        {"role": "system", "content": "You are a legal assistant."},
        {"role": "user", "content": "Can you give examples of martial law applications during different reigns in England?"},
        {"role": "assistant", "content": "During the reign of Queen Mary, a proclamation was issued stating that the possession of heretical and seditious works would be punished by martial law. Similarly, in the reign of Queen Elizabeth I, those involved in the rebellion led by the Earls of Northumberland and Westmoreland were executed under martial law."},
        {"role": "user", "content": "Were there any legal objections to such use of martial law?"},
        {"role": "assistant", "content": "Yes, Lord Coke stated that executing someone under martial law in times of peace was against Magna Carta and thus considered murder. This reflects the legal contention that martial law should not be applied when normal judicial processes are available."}
    ]

    chat_messages_4 = [
        {"role": "system", "content": "You are a legal assistant."},
        {"role": "user", "content": "Who has the power to proclaim martial law in England?"},
        {"role": "assistant", "content": "The power to proclaim martial law in England was traditionally vested in the Crown. However, this power has been subject to legal limitations, such as those imposed by the Petition of Right in 1628, which restricted the Crown's ability to issue such proclamations without Parliament's consent."},
        {"role": "user", "content": "What were the arguments for and against the Crown's power to proclaim martial law?"},
        {"role": "assistant", "content": "Proponents argued that the Crown could exercise this power to avert imminent danger, as described by Finch C.J. However, critics like Mr. Holborne contended that such power should only be exercised in cases of actual and apparent danger, not on mere suspicion or fear of potential threats."}
    ]

    # Combine all chat messages into a single list
    all_chat_messages = [chat_messages_1, chat_messages_2, chat_messages_3, chat_messages_4]

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(all_chat_messages, test_size=0.15, random_state=42)

    # Generate and print the response for training data
    logging.info("Generating response for training data...")
    train_response = generate_response(train_data[0])
    print("Training Response:", train_response)

    # Validate the model using validation data
    logging.info("Validating the model...")
    validation_results = validate_model(val_data)
    for i, result in enumerate(validation_results):
        print(f"Validation Response {i+1}:", result)

if __name__ == "__main__":
    main()
