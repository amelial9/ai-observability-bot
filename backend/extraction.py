import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os


EXCEL_PATH = "../Pemco_faqs.xlsx"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
CHROMA_COLLECTION_NAME = "company_faqs"
CHROMA_PERSIST_PATH = "my_chroma_db"

# 1. Load data from Excel
def load_faqs_from_excel(excel_path):
    """
    Loads FAQ data from an Excel file, assuming columns named 'Question' and 'Answer'.
    Combines Q&A into 'content_to_embed' for embedding.
    """
    try:
        df = pd.read_excel(excel_path)
        faqs = []
        for index, row in df.iterrows():
            question = row.get("Question", "").strip()
            answer = row.get("Answer", "").strip()

            if question and answer: # Only add if both question and answer are present
                faqs.append({
                    "question": question,
                    "answer": answer,
                    "content_to_embed": f"Question: {question}\nAnswer: {answer}"
                })
            else:
                print(f"Warning: Skipping row {index+2} due to missing Question or Answer.") # +2 for 0-indexed row and header

        if not faqs:
            print(f"Warning: No valid FAQ data found in '{excel_path}'. Check column names and content.")
            return None
        return faqs
    except FileNotFoundError:
        print(f"Error: The Excel file '{excel_path}' was not found.")
        return None
    except KeyError as e:
        print(f"Error: Missing expected column in '{excel_path}'. Please ensure 'Question' and 'Answer' columns exist. Details: {e}")
        return None


# 2. Initialize Embedding Model and Vector Database
def initialize_rag_components():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"Error loading embedding model. Please ensure 'sentence-transformers' is installed correctly and you have an internet connection for the first download: {e}")
        return None, None

    print(f"Initializing ChromaDB PersistentClient at: {CHROMA_PERSIST_PATH}...")
    try:
        # Ensure the directory exists. os.makedirs creates it if it doesn't.
        os.makedirs(CHROMA_PERSIST_PATH, exist_ok=True)
        
        # Connect to the persistent ChromaDB client
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
        
        # Get or create the collection. If it exists, it will load. If not, it will be created.
        collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        print(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' ready and connected.")
        return embedding_model, collection
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        print(f"Please check write permissions for '{CHROMA_PERSIST_PATH}' and ensure ChromaDB dependencies are met.")
        return None, None # Return None for both if DB initialization fails

# 3. Embed and Store Data
def index_faqs_to_chroma(faqs_data, embedding_model, collection):
    """
    Embeds the FAQ data and stores it in the ChromaDB collection.
    Includes logic to clear existing data to prevent duplicates on re-run.
    """
    if faqs_data is None or not faqs_data: # Check if data was loaded successfully and is not empty
        print("No valid FAQ data to index. Aborting indexing process.")
        return

    print("Indexing FAQs into vector database...")
    
    documents = [faq['content_to_embed'] for faq in faqs_data]
    metadatas = [{"question": faq['question'], "answer": faq['answer']} for faq in faqs_data]
    ids = [f"faq_{i}" for i in range(len(faqs_data))]

    # Retrieve all existing IDs in the collection
    try:
        all_current_ids_in_collection = collection.get()['ids']
        if all_current_ids_in_collection:
            print(f"Clearing {len(all_current_ids_in_collection)} existing entries from collection before re-indexing.")
            collection.delete(ids=all_current_ids_in_collection)
        else:
            print("ChromaDB collection is empty. No existing data to clear.")
    except Exception as e:
        print(f"Warning: Could not clear existing data from ChromaDB. Error: {e}")
        print("This might be a fresh collection or an issue with the .get() method.")

    print(f"Generating embeddings for {len(documents)} documents...")
    try:
        embeddings = embedding_model.encode(documents).tolist()
        print("Embeddings generated successfully.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        print("This might be due to an issue with the embedding model or input data.")
        return

    print(f"Adding {len(documents)} documents to ChromaDB collection...")
    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        print(f"Successfully indexed {len(faqs_data)} FAQs into ChromaDB.")
    except Exception as e:
        print(f"Error adding documents to ChromaDB: {e}")
        print("Please check your data format and ChromaDB connection.")


if __name__ == "__main__":
    print("--- Starting FAQ Indexing Script ---")
    
    faqs_data = load_faqs_from_excel(EXCEL_PATH)
    
    if faqs_data is not None:
        embedding_model, collection = initialize_rag_components()
        
        if embedding_model is not None and collection is not None: 
            index_faqs_to_chroma(faqs_data, embedding_model, collection)
            print("\n--- Indexing Process Complete ---")
            print(f"Your vector database is now persisted on disk at: '{CHROMA_PERSIST_PATH}'")
            print("You can now use your BeeAI agent script to connect to this persistent database.")
        else:
            print("\n--- Indexing Process Aborted ---")
            print("Failed to initialize RAG components. Please check previous error messages.")
    else:
        print("\n--- Indexing Process Aborted ---")
        print("Failed to load FAQ data from Excel. Please check previous error messages.")