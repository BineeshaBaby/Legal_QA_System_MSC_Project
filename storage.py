import os
import json
import openai
import logging
from sentence_transformers import SentenceTransformer
from chromadb import Client as ChromaClient
from chromadb.config import Settings
import warnings

# Suppress specific warnings
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
    Stores embeddings in ChromaDB.

    Args:
        collection (Collection): ChromaDB collection object.
        embeddings (list): List of embeddings generated for text chunks.
        text_chunks (list): List of text chunks.
        metadata_list (list): List of metadata dictionaries for each chunk.
    """
    logging.info("Storing embeddings in ChromaDB...")

    # Generate unique IDs for each document using a simple numbering scheme
    ids = [f"doc_{i}" for i in range(len(text_chunks))]  
    
    def sanitize_metadata(metadata):
        """
        Ensures metadata values are of allowed types for storage.

        Args:
            metadata (dict): Metadata dictionary to sanitize.

        Returns:
            dict: Sanitized metadata dictionary.
        """
        sanitized_metadata = {}
        # Iterate over the metadata dictionary
        for key, value in metadata.items():
            # Convert lists and dictionaries to strings to avoid storage issues
            if isinstance(value, (list, dict)):
                sanitized_metadata[key] = str(value)
            else:
                sanitized_metadata[key] = value
        return sanitized_metadata

    # Iterate over the text chunks, embeddings, metadata, and generated IDs
    for text, embedding, metadata, doc_id in zip(text_chunks, embeddings, metadata_list, ids):
        sanitized_metadata = sanitize_metadata(metadata)  # Sanitize the metadata
        # Store each chunk with its associated metadata and embedding in ChromaDB
        collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[sanitized_metadata],
            embeddings=[embedding]
        )
    logging.info("Embeddings successfully stored in ChromaDB.")

def load_text_chunks_and_metadata(text_chunks_file, metadata_list_file):
    """
    Loads text chunks and metadata from files.

    Args:
        text_chunks_file (str): Path to the text chunks file.
        metadata_list_file (str): Path to the metadata file.

    Returns:
        tuple: A tuple containing the list of text chunks and the list of metadata dictionaries.
    """
    logging.info("Loading text chunks and metadata from files...")

    # Read the text chunks from the file, stripping any extra whitespace
    with open(text_chunks_file, "r", encoding="utf-8") as f:
        text_chunks = [line.strip() for line in f]

    # Load the metadata from the JSON file
    with open(metadata_list_file, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    logging.info("Text chunks and metadata loaded successfully.")
    return text_chunks, metadata_list

def query_chromadb(collection, query_text, top_k=5):
    """
    Queries ChromaDB with a given text.

    Args:
        collection (Collection): ChromaDB collection object.
        query_text (str): Text to query the database with.
        top_k (int): Number of top results to return.

    Returns:
        tuple: A tuple containing lists of documents, distances, and metadata corresponding to the query results.
    """
    try:
        # Create the query embedding using the model
        query_embedding = model.encode([query_text])[0].tolist()
        
        # Perform the query using the embedding, retrieving the top_k results
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Return the documents, distances, and metadata from the query results
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
    # Create embeddings for each chunk using the Sentence Transformer model
    embeddings = model.encode(text_chunks).tolist()
    logging.info("Embeddings created successfully.")

    # Store embeddings in ChromaDB
    store_embeddings_in_chromadb(collection, embeddings, text_chunks, metadata_list)
    logging.info("Storing in ChromaDB completed successfully.")

    # Define the query text
    query_text = "What is the fountain of marshal law?"

    # Perform the query against ChromaDB
    results = query_chromadb(collection, query_text, top_k=5)

    # Print results for manual inspection
    if results:
        documents, distances, metadatas = results
        seen_chunks = set()  # Track seen chunks to avoid duplication
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


