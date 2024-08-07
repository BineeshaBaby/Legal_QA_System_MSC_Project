import os
import openai
from dotenv import load_dotenv
import logging
import json
import time
from transformers import pipeline
import textstat
import matplotlib.pyplot as plt

# Load environment variables from a .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_knowledge_base(file_path):
    """Load the knowledge base from a specified file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the file: {file_path}")
        return {}
    except Exception as e:
        logging.error(f"Error loading knowledge base: {e}")
        return {}

def generate_response_gpt(model_name, context, question):
    """Generate a response using a specified GPT model based on the provided context and question."""
    full_prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer strictly using the provided context:"

    chat_messages = [
        {"role": "system", "content": "You are a legal assistant. Only use the provided context to answer the questions."},
        {"role": "user", "content": full_prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=chat_messages,
            max_tokens=1500,
            temperature=0.7
        )
        return response.choices[0]['message']['content']
    except openai.error.OpenAIError as e:
        logging.error(f"An error occurred: {e}")
        return "An error occurred while generating the response."

def compare_models(questions, context):
    """Compare responses from GPT-4 and GPT-3.5-turbo models for a list of questions."""
    gpt4_responses = []
    gpt35_responses = []

    for question in questions:
        gpt4_responses.append(generate_response_gpt("gpt-4", context, question))
        gpt35_responses.append(generate_response_gpt("gpt-3.5-turbo", context, question))

    return gpt4_responses, gpt35_responses

def measure_response_time(model_name, context, question):
    start_time = time.time()
    response = generate_response_gpt(model_name, context, question)
    end_time = time.time()
    return response, end_time - start_time

def sentiment_analysis(responses):
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiments = [sentiment_pipeline(response)[0] for response in responses]
    return sentiments

def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words)

def readability_analysis(text):
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    smog_index = textstat.smog_index(text)
    return flesch_reading_ease, smog_index

if __name__ == "__main__":
    from plots import plot_comparison, plot_response_time, plot_sentiment_analysis, plot_lexical_diversity, plot_readability

    knowledge_base_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\summarized_context.json"
    knowledge_base = load_knowledge_base(knowledge_base_file_path)

    if not knowledge_base:
        logging.error("Knowledge base is empty or could not be loaded.")
        exit()

    context = knowledge_base.get("summarized_context", "")

    test_questions = [
        "How does common law support actions taken during martial law?",
        "What is martial law?",
        "How does common law support actions taken during martial law?",
        "What did the Petition of Right of 1628 assert regarding martial law?",
        "What restrictions did the Petition of Right of 1628 place on the Crown's power to declare martial law?",
        "What legal measures were implemented in the UK during World War I and II to address national security threats?",
        "How does Indian law address the release of films and the protection of copyrights in databases and software?",
        "How has the Indian judiciary maintained a fair environment for the commercialization of IPR?"
    ]

    gpt4_responses, gpt35_responses = compare_models(test_questions, context)

    for i, question in enumerate(test_questions):
        print(f"Question: {question}")
        print(f"GPT-4 Response: {gpt4_responses[i]}")
        print(f"GPT-3.5-turbo Response: {gpt35_responses[i]}")
        print("\n")

    gpt4_scores = [
        [4.8, 4.5, 4.7, 4.6],  # Scores for Q1
        [4.7, 4.4, 4.8, 4.7],  # Scores for Q2
        [4.8, 4.5, 4.7, 4.6],  # Scores for Q3
        [4.7, 4.4, 4.8, 4.7]   # Scores for Q4
    ]

    gpt35_scores = [
        [3.8, 4.3, 3.6, 3.4],  # Scores for Q1
        [3.4, 3.2, 3.5, 3.3],  # Scores for Q2
        [3.5, 3.3, 3.6, 3.4],  # Scores for Q3
        [3.4, 3.2, 3.5, 3.3]   # Scores for Q4
    ]

    criteria = ['Relevance', 'Accuracy', 'Coherence', 'Completeness']
    plot_comparison(gpt4_scores, gpt35_scores, criteria)

    # Additional analyses
    gpt4_response, gpt4_time = measure_response_time("gpt-4", context, "What is martial law?")
    gpt35_response, gpt35_time = measure_response_time("gpt-3.5-turbo", context, "What is martial law?")
    print(f"GPT-4 Response Time: {gpt4_time} seconds")
    print(f"GPT-3.5-turbo Response Time: {gpt35_time} seconds")

    gpt4_sentiments = sentiment_analysis(gpt4_responses)
    gpt35_sentiments = sentiment_analysis(gpt35_responses)
    print(f"GPT-4 Sentiments: {gpt4_sentiments}")
    print(f"GPT-3.5-turbo Sentiments: {gpt35_sentiments}")

    gpt4_diversity = lexical_diversity(gpt4_responses[0])
    gpt35_diversity = lexical_diversity(gpt35_responses[0])
    print(f"GPT-4 Lexical Diversity: {gpt4_diversity}")
    print(f"GPT-3.5-turbo Lexical Diversity: {gpt35_diversity}")

    gpt4_readability = readability_analysis(gpt4_responses[0])
    gpt35_readability = readability_analysis(gpt35_responses[0])
    print(f"GPT-4 Readability: {gpt4_readability}")
    print(f"GPT-3.5-turbo Readability: {gpt35_readability}")
    
    # Plot each comparison
    plot_response_time(gpt4_time, gpt35_time)
    plot_sentiment_analysis(gpt4_sentiments + gpt35_sentiments)  # Plotting combined sentiment analysis for simplicity
    plot_lexical_diversity([gpt4_diversity, gpt35_diversity])
    plot_readability(gpt35_readability[0], gpt35_readability[1])
