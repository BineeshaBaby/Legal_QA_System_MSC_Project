import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(gpt4_scores, gpt35_scores, criteria):
    """Plot a radar chart comparing the average scores of GPT-4 and GPT-3.5-turbo."""
    gpt4_avg_scores = np.mean(gpt4_scores, axis=0)
    gpt35_avg_scores = np.mean(gpt35_scores, axis=0)

    num_vars = len(criteria)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    gpt4_avg_scores = np.append(gpt4_avg_scores, gpt4_avg_scores[0])
    gpt35_avg_scores = np.append(gpt35_avg_scores, gpt35_avg_scores[0])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, gpt4_avg_scores, color='blue', alpha=0.25)
    ax.fill(angles, gpt35_avg_scores, color='red', alpha=0.25)

    ax.plot(angles, gpt4_avg_scores, color='blue', linewidth=2, label='GPT-4')
    ax.plot(angles, gpt35_avg_scores, color='red', linewidth=2, label='GPT-3.5-turbo')

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    plt.title('Performance Comparison: GPT-4 vs GPT-3.5-turbo')
    plt.show()

def plot_response_time(gpt4_time, gpt35_time):
    models = ['GPT-4', 'GPT-3.5-turbo']
    times = [gpt4_time, gpt35_time]
    plt.figure(figsize=(10, 5))
    plt.bar(models, times, color=['blue', 'red'])
    plt.xlabel('Models')
    plt.ylabel('Response Time (seconds)')
    plt.title('Response Time Comparison')
    plt.show()

def plot_sentiment_analysis(sentiments):
    sentiment_labels = [sentiment['label'] for sentiment in sentiments]
    sentiment_scores = [sentiment['score'] for sentiment in sentiments]
    questions = [f"Q{i+1}" for i in range(len(sentiments))]
    
    colors = ['green' if label == 'POSITIVE' else 'red' for label in sentiment_labels]
    
    plt.figure(figsize=(12, 6))
    plt.bar(questions, sentiment_scores, color=colors)
    plt.xlabel('Questions')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Analysis')
    plt.ylim(0, 1)
    plt.show()

def plot_lexical_diversity(diversity):
    models = ['GPT-4', 'GPT-3.5-turbo']
    plt.figure(figsize=(10, 5))
    plt.bar(models, diversity, color=['blue', 'red'])
    plt.xlabel('Models')
    plt.ylabel('Lexical Diversity')
    plt.title('Lexical Diversity Comparison')
    plt.ylim(0, 1)
    plt.show()

def plot_readability(flesch_reading_ease, smog_index):
    labels = ['Flesch Reading Ease', 'SMOG Index']
    scores = [flesch_reading_ease, smog_index]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, scores, width, color=['blue', 'orange'])

    ax.set_xlabel('Readability Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Readability Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(['GPT-4', 'GPT-3.5-turbo'])

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom')

    plt.show()
