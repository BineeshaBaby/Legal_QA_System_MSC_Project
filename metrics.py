import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import pipeline
import time

def calculate_bleu(reference, generated):
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing_function)
    return bleu_score

def calculate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def measure_latency(model_func, context, question):
    start_time = time.time()
    response = model_func(context, question)
    end_time = time.time()
    latency = end_time - start_time
    return latency, response

# Sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(response):
    """Analyze the sentiment of a generated response."""
    sentiment = sentiment_pipeline(response)[0]['label']
    return sentiment

def calculate_lexical_diversity(response):
    """Calculate the lexical diversity of a generated response."""
    words = response.split()
    lexical_diversity = len(set(words)) / len(words)
    return lexical_diversity


