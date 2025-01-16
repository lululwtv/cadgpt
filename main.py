import bs4
import os
import getpass

from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

from typing_extensions import List, TypedDict

from dotenv import load_dotenv
load_dotenv()
 
FILE_PATH = os.getenv('FILE_PATH')

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

# Ensure USER_AGENT is set
if 'USER_AGENT' not in os.environ:
    os.environ['USER_AGENT'] = "cadgpt"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"
# os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Ensure your VertexAI credentials are configured
model = ChatVertexAI(model="gemini-1.5-flash")

# Initialize VertexAI Embeddings
embeddings = VertexAIEmbeddings(model="text-embedding-004")

# Create an in-memory vector store
vector_store = InMemoryVectorStore(embeddings)

# Load and chunk contents of the blog
pages = load_documents()
chunks = split_documents(pages)
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
docs = vector_store.similarity_search("what is a 3d construction primitive function", k=5)

# Print the top 5 most similar documents
for doc in docs:
    print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')

# Tavily search
searchTool = TavilySearchResults(query="what is a 3d construction primitive function")

# Create toolset
tools = [searchTool]

# Invoke the tools before chatting with LLM
model_tools = model.bind_tools(tools)

# Chat with LLM
response = model.invoke([HumanMessage(content="what are you capable of?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")