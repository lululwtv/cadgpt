import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.schema.document import Document
from langchain_chroma import Chroma
import chromadb
from embeddings import get_embedding_function

from dotenv import load_dotenv
load_dotenv()
 
FILE_PATH = os.getenv('FILE_PATH')
CHROMA_COLLECTION = os.getenv('CHROMA_COLLECTION')
CHROMA_PATH = os.getenv('CHROMA_PATH')


# Ensure USER_AGENT is set
if 'USER_AGENT' not in os.environ:
    os.environ['USER_AGENT'] = "cadgpt"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"


def main():
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    pages = load_documents()
    chunks = split_documents(pages)
    add_to_chroma(chunks)

def load_documents():
    loader = PyPDFLoader(FILE_PATH)
    pages = loader.load_and_split()
    return pages

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    persistent_client = chromadb.PersistentClient()
    # Load the existing database.
    vector_store = Chroma(
        client=persistent_client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = vector_store.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vector_store.add_documents(new_chunks, ids=new_chunk_ids)
        # vector_store.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()