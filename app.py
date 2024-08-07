import os
import openai
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from chat_model import load_knowledge_base, generate_response


# Load environment variables from a .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    document: Optional[str] = None  # Optional document name

@app.post("/query")
async def query_legal_documents(request: QueryRequest):
    question = request.question
    document = request.document
    context_chunks = []

    # Load the processed chunks from JSON file
    processed_chunks_file = r'C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_text_chunks.json'
    processed_chunks = load_knowledge_base(processed_chunks_file)

    if not processed_chunks:
        raise HTTPException(status_code=500, detail="Failed to load the knowledge base.")

    if document:
        document_chunks = [item for item in processed_chunks if item.get("document_id") == document]
        if document_chunks:
            context_chunks = document_chunks
        else:
            return {"question": question, "answer": f"Document '{document}' not found."}
    else:
        context_chunks = processed_chunks

    # Generate responses from each relevant chunk
    answers = []
    for chunk in context_chunks:
        context = chunk.get("enhanced", "")
        if context:
            answer = generate_response(context, question)
            if "An error occurred" not in answer:
                answers.append(answer)

    if answers:
        return {"question": question, "answer": " ".join(answers)}
    else:
        return {"question": question, "answer": "The question is outside the context of the provided documents. Please provide more specific details or context."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)

# To run the FastAPI app, use the command: uvicorn app:app --reload --port 8080
