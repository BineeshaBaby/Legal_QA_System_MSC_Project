# app/query.py
from transformers import AutoTokenizer, AutoModel
import numpy as np
import openai

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def vectorize_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy().astype(np.float32)
    return embeddings

def retrieve_documents(query, collection, k=5):
    query_vector = vectorize_text(query)
    results = collection.query(query_vector.tolist())

    if not results['documents'][0]:
        raise ValueError("No relevant documents found in ChromaDB.")
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    combined_results = [{'text': doc, 'metadata': meta, 'distance': dist} 
                        for doc, meta, dist in zip(documents, metadatas, distances)]
    
    combined_results = sorted(combined_results, key=lambda x: x['distance'])
    
    return [result['text'] for result in combined_results[:k]]

def get_answer(query, collection):
    retrieved_docs = retrieve_documents(query, collection)
    context = "\n\n".join(retrieved_docs)
    rag_query = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    
    response = openai.Completion.create(
        engine="text-davinci-003",  # Adjust engine as per your requirements
        prompt=rag_query,
        max_tokens=1000
    )
    
    return response.choices[0].text.strip() if response.choices else "No answer found."

