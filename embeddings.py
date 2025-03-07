from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import chromadb.utils.embedding_functions as embedding_functions
from langchain_openai import OpenAIEmbeddings
import openai 

def get_embedding_function():
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-ada-002"
    # )
    # Initialize VertexAI Embeddings
    embeddings = VertexAIEmbeddings(model="text-embedding-004")
    return embeddings
