from fastapi import FastAPI
from pydantic import BaseModel 
import os
import openai
from dotenv import load_dotenv 
from chat_model import load_knowledge_base, generate_response_gpt4


# Load environment variables from a .env file
load_dotenv()  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY  

# Initialize FastAPI app
backend = FastAPI()  # Creates an instance of FastAPI, which will be used to define routes and handle HTTP requests.

# Define the request model
class QuestionRequest(BaseModel):
    question: str  # Defines a Pydantic model for validating incoming JSON requests, with a single field `question` of type string.

# Load the summarized context
knowledge_base_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\summarized_context.json"
# Specifies the path to the JSON file containing the summarized context.

context = load_knowledge_base(knowledge_base_file_path)
# Loads the summarized context from the specified file using the `load_knowledge_base` function.

# Define the FastAPI endpoint
@backend.post("/ask")  # Defines an endpoint at the URL path "/ask" that accepts POST requests.
def ask_question(request: QuestionRequest):
    """
    Handles POST requests to the /ask endpoint.

    Args:
        request (QuestionRequest): The incoming request containing the user's question.

    Returns:
        dict: A dictionary containing the generated answer to the user's question.
    """
    answer = generate_response_gpt4(context, request.question)
    # Generates a response using the GPT-4 model based on the loaded context and the user's question.

    return {"answer": answer}  # Returns the generated answer as a JSON response.

# uvicorn backend:backend --reload
# Command to run the FastAPI app with uvicorn, which is a lightning-fast ASGI server.
# `backend:backend` indicates that uvicorn should look for an app instance named `backend` in a file named `backend.py`.
# `--reload` enables auto-reloading, so the server will restart whenever the code changes.


