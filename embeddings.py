from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import chromadb.utils.embedding_functions as embedding_functions

def get_embedding_function():
    # Instruct Embeddings
    # embeddings = embedding_functions.InstructorEmbeddingFunction(
    #     model_name="hkunlp/instructor-base", 
    #     device="cuda"
    # )
    
    # Initialize VertexAI Embeddings
    embeddings = VertexAIEmbeddings(model="text-embedding-004")
    return embeddings
