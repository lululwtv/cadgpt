from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
import os
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Ensure USER_AGENT is set
if 'USER_AGENT' not in os.environ:
    os.environ['USER_AGENT'] = "myagent"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

# Ensure your VertexAI credentials are configured
llm = ChatVertexAI(model="gemini-1.5-flash")

# Initialize VertexAI Embeddings
embeddings = VertexAIEmbeddings(model="text-embedding-004")

# Create an in-memory vector store
vector_store = InMemoryVectorStore(embeddings)

# Load and chunk contents of the blog
file_path = "./documents/CadQuery Cheatsheet.pdf"
loader = PyPDFLoader("./documents/CadQuery Cheatsheet.pdf")
pages = []
for page in loader.load():
    pages.append(page)

vector_store = InMemoryVectorStore.from_documents(pages, embeddings)
docs = vector_store.similarity_search("what is this about")

for doc in docs:
    print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')