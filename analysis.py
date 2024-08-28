import os 
import openai  
import logging  
from dotenv import load_dotenv  
import matplotlib.pyplot as plt  
import numpy as np

# Import custom functions from other modules
from chat_model import generate_response_gpt4
from metrics import (
    calculate_bleu, 
    calculate_rouge, 
    measure_latency, 
    analyze_sentiment, 
    calculate_lexical_diversity
)

# Load environment variables from a .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Retrieve the OpenAI API key from environment variables

# Set the OpenAI API key for use in API calls
openai.api_key = OPENAI_API_KEY

# Configure logging to display info-level messages with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_response_gpt3(context, question):
    """
    Generate a response using GPT-3.5-turbo based on the provided context and question.
    
    Args:
        context (str): The context to provide to the model.
        question (str): The user's question.
    
    Returns:
        str: The generated response from GPT-3.5-turbo.
    """
    try:
        # Request a response from GPT-3.5-turbo
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            max_tokens=1500,  # Limit the response length
            temperature=0.7  # Control the creativity of the response
        )
        return response.choices[0]['message']['content']  # Extract and return the content of the response
    except openai.error.OpenAIError as e:
        logging.error(f"An error occurred with GPT-3: {e}")  # Log any errors that occur
        return None  # Return None if an error occurs

def compare_models(context, question, reference_answer):
    """
    Compare responses from GPT-4 and GPT-3.5-turbo models.
    
    Args:
        context (str): The context to provide to the models.
        question (str): The user's question.
        reference_answer (str): The reference answer for evaluation.
    
    Returns:
        dict: A dictionary containing comparison results for both models.
    """
    results = {}  # Initialize a dictionary to store results

    # Measure performance of GPT-4
    latency_gpt4, response_gpt4 = measure_latency(generate_response_gpt4, context, question)
    bleu_gpt4 = calculate_bleu(reference_answer, response_gpt4)
    rouge_gpt4 = calculate_rouge(reference_answer, response_gpt4)
    sentiment_gpt4 = analyze_sentiment(response_gpt4)
    diversity_gpt4 = calculate_lexical_diversity(response_gpt4)

    # Store GPT-4 results in the dictionary
    results['GPT-4'] = {
        'latency': latency_gpt4,
        'bleu': bleu_gpt4,
        'rouge': rouge_gpt4,
        'sentiment': sentiment_gpt4,
        'lexical_diversity': diversity_gpt4,
        'response': response_gpt4
    }

    # Measure performance of GPT-3.5-turbo
    latency_gpt3, response_gpt3 = measure_latency(generate_response_gpt3, context, question)
    bleu_gpt3 = calculate_bleu(reference_answer, response_gpt3)
    rouge_gpt3 = calculate_rouge(reference_answer, response_gpt3)
    sentiment_gpt3 = analyze_sentiment(response_gpt3)
    diversity_gpt3 = calculate_lexical_diversity(response_gpt3)

    # Store GPT-3.5-turbo results in the dictionary
    results['GPT-3.5-turbo'] = {
        'latency': latency_gpt3,
        'bleu': bleu_gpt3,
        'rouge': rouge_gpt3,
        'sentiment': sentiment_gpt3,
        'lexical_diversity': diversity_gpt3,
        'response': response_gpt3
    }

    # Print results for GPT-4
    print("GPT-4 Results:")
    print(f"Latency: {latency_gpt4:.2f} seconds")
    print(f"BLEU Score: {bleu_gpt4:.4f}")
    print(f"ROUGE Scores: {rouge_gpt4}")
    print(f"Sentiment: {sentiment_gpt4}")
    print(f"Lexical Diversity: {diversity_gpt4:.4f}")
    print(f"Generated Response: {response_gpt4}")

    # Print results for GPT-3.5-turbo
    print("\nGPT-3 Results:")
    print(f"Latency: {latency_gpt3:.2f} seconds")
    print(f"BLEU Score: {bleu_gpt3:.4f}")
    print(f"ROUGE Scores: {rouge_gpt3}")
    print(f"Sentiment: {sentiment_gpt3}")
    print(f"Lexical Diversity: {diversity_gpt3:.4f}")
    print(f"Generated Response: {response_gpt3}")

    return results  # Return the comparison results

# Example context (summarized for brevity)
context = """
Common law has a historical basis in maintaining order during times of martial law. Martial law refers to the 
temporary imposition of direct military control of normal civil functions or suspension of civil law by a government, 
especially in response to a temporary emergency where civil forces are overwhelmed. 
The Petition of Right of 1628 was a significant constitutional document that restricted the Crown's ability to 
impose martial law arbitrarily. It asserted the rights and liberties of subjects, particularly concerning unlawful 
imprisonment and the application of martial law. 
During World War I and II, the UK implemented several legal measures to address national security threats, which 
included the Defense of the Realm Acts and other regulations to control various aspects of life and ensure public safety. 
Indian law, on the other hand, has its framework addressing intellectual property rights (IPR) protection, including 
measures for the release of films and protection of copyrights in databases and software. The Indian judiciary has played 
a crucial role in maintaining a fair environment for the commercialization of IPR.
"""

# List of questions related to the context
questions = [
    "How does common law support actions taken during martial law?",
    "What is martial law?",
    "What did the Petition of Right of 1628 assert regarding martial law?",
    "What restrictions did the Petition of Right of 1628 place on the Crown's power to declare martial law?",
    "What legal measures were implemented in the UK during World War I and II to address national security threats?",
    "How does Indian law address the release of films and the protection of copyrights in databases and software?",
    "How has the Indian judiciary maintained a fair environment for the commercialization of IPR?"
]

# Reference answers for evaluation (ideally created by a subject matter expert)
reference_answers = [
    "Common law supports actions during martial law by providing a legal framework that upholds civil rights while allowing necessary measures during emergencies.",
    "Martial law is the imposition of direct military control over normal civilian functions or suspension of civil law by a government.",
    "The Petition of Right of 1628 asserted that the Crown could not impose martial law arbitrarily and must respect the liberties of subjects.",
    "The Petition of Right of 1628 restricted the Crown's power by asserting that martial law could not be imposed without the consent of Parliament.",
    "During World War I and II, the UK implemented the Defense of the Realm Acts and other regulations to control various aspects of life and ensure public safety.",
    "Indian law protects the release of films and copyrights in databases and software through its intellectual property rights framework.",
    "The Indian judiciary has maintained a fair environment for the commercialization of IPR by enforcing laws that protect intellectual property rights."
]

# Store the comparison results for all questions
all_results = []

# Compare GPT-4 and GPT-3.5-turbo on each question
for i, question in enumerate(questions):
    print(f"\nQuestion {i+1}: {question}")
    result = compare_models(context, question, reference_answers[i])
    all_results.append(result)

# Initialize lists to store the extracted data for each metric
latencies_gpt4 = []
latencies_gpt35 = []
bleu_scores_gpt4 = []
bleu_scores_gpt35 = []
rouge1_scores_gpt4 = []
rouge1_scores_gpt35 = []
rouge2_scores_gpt4 = []
rouge2_scores_gpt35 = []
rougeL_scores_gpt4 = []
rougeL_scores_gpt35 = []
sentiments_gpt4 = []
sentiments_gpt35 = []
diversity_gpt4 = []
diversity_gpt35 = []

# Extract data from all_results for plotting
for result in all_results:
    latencies_gpt4.append(result['GPT-4']['latency'])
    latencies_gpt35.append(result['GPT-3.5-turbo']['latency'])
    bleu_scores_gpt4.append(result['GPT-4']['bleu'])
    bleu_scores_gpt35.append(result['GPT-3.5-turbo']['bleu'])
    rouge1_scores_gpt4.append(result['GPT-4']['rouge']['rouge1'])
    rouge1_scores_gpt35.append(result['GPT-3.5-turbo']['rouge']['rouge1'])
    rouge2_scores_gpt4.append(result['GPT-4']['rouge']['rouge2'])
    rouge2_scores_gpt35.append(result['GPT-3.5-turbo']['rouge']['rouge2'])
    rougeL_scores_gpt4.append(result['GPT-4']['rouge']['rougeL'])
    rougeL_scores_gpt35.append(result['GPT-3.5-turbo']['rouge']['rougeL'])
    sentiments_gpt4.append(result['GPT-4']['sentiment'])
    sentiments_gpt35.append(result['GPT-3.5-turbo']['sentiment'])
    diversity_gpt4.append(result['GPT-4']['lexical_diversity'])
    diversity_gpt35.append(result['GPT-3.5-turbo']['lexical_diversity'])

# Aggregate the results across all questions
total_latency_gpt4 = sum(latencies_gpt4)
total_latency_gpt35 = sum(latencies_gpt35)
average_bleu_gpt4 = np.mean(bleu_scores_gpt4)
average_bleu_gpt35 = np.mean(bleu_scores_gpt35)
average_rouge1_gpt4 = np.mean(rouge1_scores_gpt4)
average_rouge1_gpt35 = np.mean(rouge1_scores_gpt35)
average_rouge2_gpt4 = np.mean(rouge2_scores_gpt4)
average_rouge2_gpt35 = np.mean(rouge2_scores_gpt35)
average_rougeL_gpt4 = np.mean(rougeL_scores_gpt4)
average_rougeL_gpt35 = np.mean(rougeL_scores_gpt35)
average_diversity_gpt4 = np.mean(diversity_gpt4)
average_diversity_gpt35 = np.mean(diversity_gpt35)

# Helper function to count the number of Positive, Neutral, and Negative sentiments
def count_sentiments(sentiments):
    return [sentiments.count('POSITIVE'), sentiments.count('NEUTRAL'), sentiments.count('NEGATIVE')]

sentiment_distribution_gpt4 = count_sentiments(sentiments_gpt4)
sentiment_distribution_gpt35 = count_sentiments(sentiments_gpt35)

# Plot total response times for both models
def plot_total_latency(total_latency_gpt4, total_latency_gpt35):
    labels = ['GPT-4', 'GPT-3.5-turbo']
    times = [total_latency_gpt4, total_latency_gpt35]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, times, color=['blue', 'orange'])
    plt.ylabel('Total Response Time (seconds)', fontsize=14)
    plt.title('Comparison of Total Response Times', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

plot_total_latency(total_latency_gpt4, total_latency_gpt35)

# Plot average BLEU scores for both models
def plot_average_bleu(average_bleu_gpt4, average_bleu_gpt35):
    labels = ['GPT-4', 'GPT-3.5-turbo']
    scores = [average_bleu_gpt4, average_bleu_gpt35]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, scores, color=['blue', 'orange'])
    plt.ylabel('Average BLEU Score', fontsize=14)
    plt.title('Average BLEU Score Comparison', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

plot_average_bleu(average_bleu_gpt4, average_bleu_gpt35)

# Plot average ROUGE scores for both models
def plot_average_rouge(average_rouge1_gpt4, average_rouge1_gpt35, average_rouge2_gpt4, average_rouge2_gpt35, average_rougeL_gpt4, average_rougeL_gpt35):
    labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    gpt4_scores = [average_rouge1_gpt4, average_rouge2_gpt4, average_rougeL_gpt4]
    gpt35_scores = [average_rouge1_gpt35, average_rouge2_gpt35, average_rougeL_gpt35]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, gpt4_scores, width, label='GPT-4', color='blue')
    plt.bar(x + width/2, gpt35_scores, width, label='GPT-3.5-turbo', color='orange')
    plt.xlabel('ROUGE Metrics', fontsize=14)
    plt.ylabel('Average ROUGE Scores', fontsize=16)
    plt.title('Comparison of Average ROUGE Scores', fontsize=18)
    plt.xticks(x, labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

plot_average_rouge(
    average_rouge1_gpt4, average_rouge1_gpt35, 
    average_rouge2_gpt4, average_rouge2_gpt35, 
    average_rougeL_gpt4, average_rougeL_gpt35
)

# Plot sentiment distribution for both models
def plot_sentiment_distribution(sentiment_distribution_gpt4, sentiment_distribution_gpt35):
    labels = ['Positive', 'Neutral', 'Negative']
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, sentiment_distribution_gpt4, width, label='GPT-4', color='blue')
    plt.bar(x + width/2, sentiment_distribution_gpt35, width, label='GPT-3.5-turbo', color='orange')
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Sentiment Distribution Comparison', fontsize=16)
    plt.xticks(x, labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

plot_sentiment_distribution(sentiment_distribution_gpt4, sentiment_distribution_gpt35)

# Plot average lexical diversity for both models
def plot_average_lexical_diversity(average_diversity_gpt4, average_diversity_gpt35):
    labels = ['GPT-4', 'GPT-3.5-turbo']
    diversity = [average_diversity_gpt4, average_diversity_gpt35]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, diversity, color=['blue', 'orange'])
    plt.ylabel('Average Lexical Diversity', fontsize=14)
    plt.title('Comparison of Average Lexical Diversity', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

plot_average_lexical_diversity(average_diversity_gpt4, average_diversity_gpt35)

# Radar chart for comparing models across multiple metrics
def plot_radar_chart(average_bleu_gpt4, average_bleu_gpt35, average_rouge1_gpt4, average_rouge1_gpt35, 
                     average_rouge2_gpt4, average_rouge2_gpt35, average_rougeL_gpt4, average_rougeL_gpt35, 
                     average_diversity_gpt4, average_diversity_gpt35, total_latency_gpt4, total_latency_gpt35):
    labels = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Lexical Diversity', 'Latency']
    
    # Normalize latency scores to fit within the chart range (0 to 1) if necessary
    max_latency = max(total_latency_gpt4, total_latency_gpt35)
    normalized_latency_gpt4 = total_latency_gpt4 / max_latency
    normalized_latency_gpt35 = total_latency_gpt35 / max_latency

    gpt4_scores = [
        average_bleu_gpt4, average_rouge1_gpt4, average_rouge2_gpt4, 
        average_rougeL_gpt4, average_diversity_gpt4, normalized_latency_gpt4
    ]
    gpt35_scores = [
        average_bleu_gpt35, average_rouge1_gpt35, average_rouge2_gpt35, 
        average_rougeL_gpt35, average_diversity_gpt35, normalized_latency_gpt35
    ]
    
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The radar chart is a complete loop, so append the start value to the end.
    gpt4_scores += gpt4_scores[:1]
    gpt35_scores += gpt35_scores[:1]
    angles += angles[:1]

    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw one line per model
    ax.fill(angles, gpt4_scores, color='blue', alpha=0.25)
    ax.fill(angles, gpt35_scores, color='orange', alpha=0.25)
    ax.plot(angles, gpt4_scores, color='blue', linewidth=2, label='GPT-4')
    ax.plot(angles, gpt35_scores, color='orange', linewidth=2, label='GPT-3.5-turbo')

    # Fix the axis to go in the right order and the correct direction
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw labels
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=14)

    # Adjust the title position with padding
    plt.title("Performance Comparison: GPT-4 vs GPT-3.5-turbo", fontsize=20, pad=30)

    # Move the legend to a new position (outside the plot area)
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=12, frameon=True, borderpad=1.5, edgecolor='black')

    # Show the plot
    plt.show()

# Call the radar chart plotting function with the aggregated results
plot_radar_chart(
    average_bleu_gpt4, average_bleu_gpt35, 
    average_rouge1_gpt4, average_rouge1_gpt35, 
    average_rouge2_gpt4, average_rouge2_gpt35, 
    average_rougeL_gpt4, average_rougeL_gpt35, 
    average_diversity_gpt4, average_diversity_gpt35, 
    total_latency_gpt4, total_latency_gpt35
)
