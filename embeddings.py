from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

def get_embedding_function():
    # Initialize VertexAI Embeddings
    embeddings = VertexAIEmbeddings(model="text-embedding-004")
    return embeddings
