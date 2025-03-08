import argparse
import os
import shutil
import re
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
import chromadb
from embeddings import get_embedding_function
from dotenv import load_dotenv
import re
import fitz

load_dotenv()


# FILE_PATH = os.getenv('FILE_PATH')
FILE_PATH = "./documents/cadquery-stable.pdf"
CHROMA_COLLECTION_CODE = os.getenv('CHROMA_COLLECTION_CODE')
CHROMA_COLLECTION_DESC = os.getenv('CHROMA_COLLECTION_DESC')
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
    
    # Load documents
    pages = load_documents(FILE_PATH)
    
    # Split documents into code and descriptions
    code_chunks, desc_chunks = split_documents(pages)
    
    # Add to Chroma
    add_to_chroma(code_chunks, CHROMA_COLLECTION_CODE)
    add_to_chroma(desc_chunks, CHROMA_COLLECTION_DESC)

def extract_text_with_pymupdf(file_path):
    doc = fitz.open(file_path)  # Open the PDF
    pages = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")  # Extract text preserving layout
        if text:
            pages.append(Document(page_content=text, metadata={"page": page_num + 1, "source": file_path}))

    return pages

def load_documents(file_path):
    file_path = file_path.strip("'\"")
    print("File path is:", file_path )
    if file_path.endswith('.md'):
        with open(file_path, 'r') as file:
            content = file.read()
            return [Document(page_content=content, metadata={"source": file_path})]
    elif file_path.endswith('.pdf'):
        pages = extract_text_with_pymupdf(file_path)
        return pages
        # with pdfplumber.open(file_path) as pdf:
        #     pages = []
        #     for page in pdf.pages:
        #         text = page.extract_text()
        #         if text:
        #             pages.append(Document(page_content=text, metadata={"page": page.page_number, "source": file_path}))
        #     return pages
    else:
        raise ValueError("Unsupported file format. Only .md and .pdf are supported.")

def split_documents(documents: list[Document]):
    code_chunks = []
    desc_chunks = []
    
    for doc in documents:
        text = doc.page_content
        code_blocks = extract_code_blocks(text)
        
        # Split descriptions
        desc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
            add_start_index=True,
        )
        desc_chunks.extend(desc_splitter.split_documents([doc]))
        
        # Split code blocks
        for block in code_blocks:
            code_doc = Document(page_content=block, metadata=doc.metadata)
            code_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=500,
                add_start_index=True,
            )
            code_chunks.extend(code_splitter.split_documents([code_doc]))
    
    return code_chunks, desc_chunks

def extract_code_blocks(text):
    code_blocks = []
    current_block = []
    in_code_block = False
    inside_fenced_block = False

    for line in text.splitlines():
        # Detect Markdown fenced code blocks
        if re.match(r'^\s*```', line):
            if inside_fenced_block:
                # Close the fenced code block
                code_blocks.append("\n".join(current_block))
                current_block = []
                inside_fenced_block = False
            else:
                inside_fenced_block = True
                current_block = []
            continue
        
        # Detect indented code (heuristic)
        if re.match(r'^\s{4,}', line) or inside_fenced_block:
            if not in_code_block:
                in_code_block = True
                current_block = []
            current_block.append(line)
        else:
            if in_code_block:
                in_code_block = False
                code_blocks.append("\n".join(current_block))
    
    # Handle last block
    if in_code_block or inside_fenced_block:
        code_blocks.append("\n".join(current_block))

    return code_blocks


def add_to_chroma(chunks: list[Document], collection_name):
    persistent_client = chromadb.PersistentClient()
    vector_store = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=get_embedding_function()
    )
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = vector_store.get(include=[])
    existing_ids = set(existing_items["ids"])
    
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vector_store.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()