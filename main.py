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
from langgraph.prebuilt import create_react_agent

from typing_extensions import List, TypedDict

from dotenv import load_dotenv
load_dotenv()

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
file_path = "./documents/CadQuery Cheatsheet.pdf"
loader = PyPDFLoader(file_path)
pages = []
for page in loader.load():
    pages.append(page)

vector_store = InMemoryVectorStore.from_documents(pages, embeddings)
docsTool = vector_store.similarity_search("what is a 3d construction primitive function", k=5)

# Print the top 5 most similar documents
for doc in docsTool:
    print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')

# Tavily search
searchTool = TavilySearchResults(query="what is a 3d construction primitive function")

# Create toolset
tools = [searchTool]

# Invoke the tools before chatting with LLM
agent_executor = create_react_agent(model, tools)

# Chat with LLM
response = agent_executor.invoke({"messages": [HumanMessage(content="what's the weather in sf")]})

print(response["messages"])