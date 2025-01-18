import bs4
import os
import getpass

from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from typing_extensions import List, TypedDict
from langchain_chroma import Chroma
import chromadb
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from embeddings import get_embedding_function

load_dotenv()
CHROMA_COLLECTION = os.getenv('CHROMA_COLLECTION')
FILE_PATH = os.getenv('FILE_PATH')
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Ensure USER_AGENT is set
if 'USER_AGENT' not in os.environ:
    os.environ['USER_AGENT'] = "cadgpt"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

def main():
    query_text = "How to create a cube"
    query_rag(query_text)

def query_rag(query_text:str):
    # Load Vector Store from Local ChromaDB
    persistent_client = chromadb.PersistentClient()
    vector_store = Chroma(
            client=persistent_client,
            collection_name=CHROMA_COLLECTION,
            embedding_function=get_embedding_function()
        )
    # Search the DB
    results = vector_store.similarity_search(query_text,k=2)
    context_text = ""

    for chunk in results:
        context_text = context_text + "\n\n---\n\n" + chunk.page_content

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # Send prompt into model
    model = ChatVertexAI(model="gemini-1.5-flash")
    response_text = model.invoke(prompt)

    # Print Model Response
    sources = [chunk.metadata.get("id", None) for chunk in results]
    formatted_response = f"\n\n\033[32mResponse: {response_text.content}\033[0m\n\nSources: {sources}\n\nUsage Metadata: {response_text.usage_metadata}"
    print(formatted_response)


if __name__ == "__main__":
    main()

# # Print the top 5 most similar documents
# for doc in docs:
#     print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')

# # Tavily search
# searchTool = TavilySearchResults(query="what is a 3d construction primitive function")

# # Create toolset
# tools = [searchTool]

# # Invoke the tools before chatting with LLM
# model_tools = model.bind_tools(tools)

# # Chat with LLM
# response = model.invoke([HumanMessage(content="what are you capable of?")])

# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")