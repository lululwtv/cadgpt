from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import chromadb.utils.embedding_functions as embedding_functions
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma

load_dotenv()

CHROMA_PATH = os.getenv('CHROMA_PATH')
CHROMA_COLLECTION = os.getenv('CHROMA_COLLECTION_DESC')

def get_embedding_function():
    """Returns an OpenAIEmbeddings object that can be used with Chroma"""
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )

