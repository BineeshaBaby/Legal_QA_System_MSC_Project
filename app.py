import os
import openai
import gradio as gr
from dotenv import load_dotenv
from datetime import datetime
from model import query_chromadb, collection  # Ensure your model functions and collection are accessible

# Load environment variables from a .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

def generate_response(question, context):
    """
    Generate a response using GPT-4 based on the provided question and context.
    """
    if not context:
        return "Unfortunately, the question you have asked falls outside the scope of the available documents. Please ask a question related to the legal information."

    chat_messages = [
        {"role": "system", "content": "You are a legal assistant."},
        {"role": "user", "content": question}
    ]
    for doc in context:
        chat_messages.append({"role": "assistant", "content": doc})
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=chat_messages
        )
        return response.choices[0]['message']['content']
    except openai.error.OpenAIError as e:
        return f"An error occurred: {e}"

# Function to handle the question and response generation
def legal_qa_interface(question):
    context_docs, distances, metadatas = query_chromadb(collection, question, top_k=5)
    context = [doc for doc in context_docs if doc]  # Ensure the context is not empty
    answer = generate_response(question, context)
    return answer, context_docs

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Legal Question Answering System
        Welcome to the legal question answering system. I am here to help you!
        """
    )
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Enter Your Legal Question",
                placeholder="Type your question here...",
                lines=2
            )
            submit_button = gr.Button("Get Answer")

        with gr.Column():
            answer_output = gr.Textbox(
                label="Answer",
                placeholder="The answer will appear here...",
                lines=5
            )
           
            
    
    submit_button.click(
        fn=legal_qa_interface,
        inputs=question_input,
        outputs=[answer_output]
    )

# Launch the interface
demo.launch()
