import os
import openai
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from chat_model import generate_response_gpt4
from metrics import calculate_bleu, calculate_rouge, measure_latency

# Load environment variables from a .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Load T5 Model and Tokenizer
t5_model_name = "t5-base"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# Function to generate response using T5
def generate_response_t5(context, question):
    input_text = f"Context: {context}\n\nQuestion: {question}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Compare GPT-4 with T5
def compare_models(context, question, reference_answer):
    results = {}

    # Measure GPT-4
    latency_gpt4, response_gpt4 = measure_latency(generate_response_gpt4, context, question)
    bleu_gpt4 = calculate_bleu(reference_answer, response_gpt4)
    rouge_gpt4 = calculate_rouge(reference_answer, response_gpt4)

    # Measure T5
    latency_t5, response_t5 = measure_latency(generate_response_t5, context, question)
    bleu_t5 = calculate_bleu(reference_answer, response_t5)
    rouge_t5 = calculate_rouge(reference_answer, response_t5)

    # Store results
    results['GPT-4'] = {
        'latency': latency_gpt4,
        'bleu': bleu_gpt4,
        'rouge': rouge_gpt4,
        'response': response_gpt4
    }
    results['T5'] = {
        'latency': latency_t5,
        'bleu': bleu_t5,
        'rouge': rouge_t5,
        'response': response_t5
    }

    # Print results
    print("GPT-4 Results:")
    print(f"Latency: {latency_gpt4:.2f} seconds")
    print(f"BLEU Score: {bleu_gpt4:.4f}")
    print(f"ROUGE Scores: {rouge_gpt4}")
    print(f"Generated Response: {response_gpt4}")
    
    print("\nT5 Results:")
    print(f"Latency: {latency_t5:.2f} seconds")
    print(f"BLEU Score: {bleu_t5:.4f}")
    print(f"ROUGE Scores: {rouge_t5}")
    print(f"Generated Response: {response_t5}")
    
    return results

# Example context and questions
# Example context (summarized for brevity)
context = """
Common law has a historical basis in maintaining order during times of martial law. Martial law refers to the temporary imposition of direct military control of normal civil functions or suspension of civil law by a government, especially in response to a temporary emergency where civil forces are overwhelmed. 
The Petition of Right of 1628 was a significant constitutional document that restricted the Crown's ability to impose martial law arbitrarily. It asserted the rights and liberties of subjects, particularly concerning unlawful imprisonment and the application of martial law. 
During World War I and II, the UK implemented several legal measures to address national security threats, which included the Defense of the Realm Acts and other regulations to control various aspects of life and ensure public safety. 
Indian law, on the other hand, has its framework addressing intellectual property rights (IPR) protection, including measures for the release of films and protection of copyrights in databases and software. The Indian judiciary has played a crucial role in maintaining a fair environment for the commercialization of IPR.
"""

# List of questions
questions = [
    "How does common law support actions taken during martial law?",
    "What is martial law?",
    "What did the Petition of Right of 1628 assert regarding martial law?",
    "What restrictions did the Petition of Right of 1628 place on the Crown's power to declare martial law?",
    "What legal measures were implemented in the UK during World War I and II to address national security threats?",
    "How does Indian law address the release of films and the protection of copyrights in databases and software?",
    "How has the Indian judiciary maintained a fair environment for the commercialization of IPR?"
]

# Reference answers for evaluation (These should ideally be created by a subject matter expert)
reference_answers = [
    "Common law supports actions during martial law by providing a legal framework that upholds civil rights while allowing necessary measures during emergencies.",
    "Martial law is the imposition of direct military control over normal civilian functions or suspension of civil law by a government.",
    "The Petition of Right of 1628 asserted that the Crown could not impose martial law arbitrarily and must respect the liberties of subjects.",
    "The Petition of Right of 1628 restricted the Crown's power by asserting that martial law could not be imposed without the consent of Parliament.",
    "During World War I and II, the UK implemented the Defense of the Realm Acts and other regulations to control various aspects of life and ensure public safety.",
    "Indian law protects the release of films and copyrights in databases and software through its intellectual property rights framework.",
    "The Indian judiciary has maintained a fair environment for the commercialization of IPR by enforcing laws that protect intellectual property rights."
]

# Store the comparison results
all_results = []

# Compare models on each question
for i, question in enumerate(questions):
    print(f"\nQuestion {i+1}: {question}")
    result = compare_models(context, question, reference_answers[i])
    all_results.append(result)

# Example of a plotting function
def plot_latency_comparison(results):
    models = ['GPT-4', 'T5']
    latencies = [results['GPT-4']['latency'], results['T5']['latency']]

    plt.figure(figsize=(8, 5))
    plt.bar(models, latencies, color=['blue', 'green'])
    plt.title('Latency Comparison: GPT-4 vs T5',fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Latency (seconds)', fontsize=14)
    plt.show()

# Plot the latency comparison for the first question
plot_latency_comparison(all_results[0])
