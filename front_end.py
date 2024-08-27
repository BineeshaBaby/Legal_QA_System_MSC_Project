<<<<<<< HEAD
import os
import openai
from dotenv import load_dotenv
import logging
import gradio as gr

# Load environment variables from a .env file
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
            return f.read()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return ""
    except Exception as e:
        logging.error(f"Error loading knowledge base: {e}")
        return ""

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

# Load the summarized context
knowledge_base_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\summarized_context.txt"
context = load_knowledge_base(knowledge_base_file_path)

def answer_question(Question):
    """
    Wrapper function to generate response for Gradio interface.
    
    Args:
        question (str): The user's question.
    
    Returns:
        tuple: The generated answer and the updated history.
    """
    answer = generate_response(context, Question)
    full_answer = answer
    history.append((Question, full_answer))
    return full_answer, get_history()

# Function to generate a response using predefined answers and save to history
history = []

def get_history():
    return "\n\n".join([f"Q: {q}\nA: {a}" for q, a in history])

# Custom CSS for styling
custom_css = """
body {
    background-color: grey;
    font-family: Arial, sans-serif;
}
.gradio-container {
    display: flex;
    flex-direction: row;
}
.left-container {
    flex: 3;
    padding: 20px;
}
.right-container {
    flex: 1;
    padding: 20px;
    background-color: #e0e0e0;
    border-radius: 10px;
}
.history-box {
    height: 400px;
    overflow-y: scroll;
    padding: 10px;
    background-color: grey;
}
.footer {
    margin-top: 20px;
    text-align: center;
}
"""

# Define Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter your Legal Question..."),
    outputs=[
        gr.Textbox(lines=5, label="Answer"),
        gr.HTML(label="History")
    ],
    title="LEGA: A Legal Assistant",
    description="Hello! How can I assist you with your legal question today?",  
    css=custom_css
)

if __name__ == "__main__":
=======
import gradio as gr
import requests

# FastAPI server URL (assuming it's running locally on port 8000)
API_URL = "http://127.0.0.1:8000/ask"

def answer_question(Question):
    # Send the question to the FastAPI backend
    response = requests.post(API_URL, json={"question": Question})
    
    if response.status_code == 200:
        answer = response.json().get("answer", "")
        history.append((Question, answer))
        return answer, get_history()
    else:
        return "Error: Could not retrieve answer", get_history()

history = []

def get_history():
    return "\n\n".join([f"<p><strong>Q: {q}</strong><br>A: {a}</p>" for q, a in history])

# Custom CSS for styling
custom_css = """
body {
    background-color: grey;
    font-family: Arial, sans-serif;
}
.gradio-container {
    display: flex;
    flex-direction: row;
}
.left-container {
    flex: 3;
    padding: 20px;
}
.right-container {
    flex: 1;
    padding: 20px;
    background-color: #e0e0e0;
    border-radius: 10px;
}
.history-box {
    height: 400px;
    overflow-y: scroll;
    padding: 10px;
    background-color: grey;
}
.footer {
    margin-top: 20px;
    text-align: center;
}
"""

# Define Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter your Legal Question..."),
    outputs=[
        gr.Textbox(lines=5, label="Answer"),
        gr.HTML(label="History")
    ],
    title="LEGA: A Perfect Legal Assistant",
    description="Hello! How can I assist you with your legal question today?",  
    css=custom_css
)

if __name__ == "__main__":
>>>>>>> b7bbe38 (adding files after end to end tesing)
    iface.launch()