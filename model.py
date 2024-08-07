import os
import json
import openai
import logging
from sentence_transformers import SentenceTransformer
from chromadb import Client as ChromaClient
from chromadb.config import Settings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Sentence Transformer model
logging.info("Initializing Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client and create a collection
persist_directory = "./chroma_db"
os.makedirs(persist_directory, exist_ok=True)
chroma_client = ChromaClient(Settings(persist_directory=persist_directory))

collection_name = "legal_documents"
collection = chroma_client.create_collection(name=collection_name)
logging.info("ChromaDB collection initialized.")

def store_embeddings_in_chromadb(collection, embeddings, text_chunks, metadata_list):
    """
    Store embeddings in ChromaDB.
    
    Args:
        collection: ChromaDB collection object.
        embeddings: List of embeddings.
        text_chunks: List of text chunks.
        metadata_list: List of metadata dictionaries.
    """
    logging.info("Storing embeddings in ChromaDB...")
    ids = [f"doc_{i}" for i in range(len(text_chunks))]  # Generate unique IDs for each document
    
    # Ensure metadata values are of allowed types
    def sanitize_metadata(metadata):
        sanitized_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                sanitized_metadata[key] = str(value)
            else:
                sanitized_metadata[key] = value
        return sanitized_metadata

    for text, embedding, metadata, doc_id in zip(text_chunks, embeddings, metadata_list, ids):
        sanitized_metadata = sanitize_metadata(metadata)
        collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[sanitized_metadata],
            embeddings=[embedding]
        )
    logging.info("Embeddings successfully stored in ChromaDB.")

def load_text_chunks_and_metadata(text_chunks_file, metadata_list_file):
    """
    Load text chunks and metadata from files.
    
    Args:
        text_chunks_file: Path to the text chunks file.
        metadata_list_file: Path to the metadata file.
    
    Returns:
        Tuple of text chunks and metadata list.
    """
    logging.info("Loading text chunks and metadata from files...")
    with open(text_chunks_file, "r", encoding="utf-8") as f:
        text_chunks = [line.strip() for line in f]

    with open(metadata_list_file, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    logging.info("Text chunks and metadata loaded successfully.")
    return text_chunks, metadata_list

def query_chromadb(collection, query_text, top_k=5):
    """
    Query ChromaDB with a given text.
    
    Args:
        collection: ChromaDB collection object.
        query_text: Text to query.
        top_k: Number of top results to return.
    
    Returns:
        Tuple of documents, distances, and metadata.
    """
    try:
        # Create the query embedding
        query_embedding = model.encode([query_text])[0].tolist()
        
        # Perform the query using the embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results['documents'], results['distances'], results['metadatas']
    except Exception as e:
        logging.error(f"Error querying ChromaDB: {str(e)}")
        return None

if __name__ == "__main__":
    # Path to the text chunks and metadata files
    text_chunks_file = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_text_chunks_all.txt"
    metadata_list_file = r"C:\Users\BINEESHA BABY\Desktop\MSC Project\combined_metadata_all.json"
    
    # Load the text chunks and metadata from files
    logging.info("Starting the loading process...")
    text_chunks, metadata_list = load_text_chunks_and_metadata(text_chunks_file, metadata_list_file)
    
    logging.info("Creating embeddings for each text chunk...")
    # Create embeddings for each chunk
    embeddings = model.encode(text_chunks).tolist()
    logging.info("Embeddings created successfully.")

    # Store in ChromaDB
    store_embeddings_in_chromadb(collection, embeddings, text_chunks, metadata_list)
    logging.info("Storing in ChromaDB completed successfully.")

    # Define your query
    query_text = "What is the fountain of marshal law?"

    # Perform the query
    results = query_chromadb(collection, query_text, top_k=5)

    # Print results for manual inspection
    if results:
        documents, distances, metadatas = results
        seen_chunks = set()
        for i, (doc_list, dist_list, meta_list) in enumerate(zip(documents, distances, metadatas), 1):
            for doc, dist, meta in zip(doc_list, dist_list, meta_list):
                chunk_id = meta.get('chunk_id', 'N/A')
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    print(f"Result {i}:")
                    print(f"Document ID: {meta.get('document_id', 'N/A')}")
                    print(f"Chunk ID: {chunk_id}")
                    print(f"Distance: {dist}")
                    print(f"Content: {doc[:500]}...")  # Print the first 500 characters for brevity
                    print("-" * 80)
    else:
        print("No results found.")
