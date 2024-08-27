from fastapi import FastAPI
from pydantic import BaseModel
import os
import openai
from dotenv import load_dotenv
from chat_model import load_knowledge_base , generate_response_gpt4

# Load environment variables from a .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize FastAPI app
backend = FastAPI()

# Define the request model
class QuestionRequest(BaseModel):
    question: str

# Load the summarized context
knowledge_base_file_path = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\summarized_context.json"
context = load_knowledge_base(knowledge_base_file_path)


# Define the FastAPI endpoint
@backend.post("/ask")
def ask_question(request: QuestionRequest):
    answer = generate_response_gpt4(context, request.question)
    return {"answer": answer}

## uvicorn backend:backend --reload

