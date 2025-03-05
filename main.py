import bs4
import os
import getpass
import logging

import openai
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

import nbformat as nbf

load_dotenv()
CHROMA_COLLECTION = os.getenv('CHROMA_COLLECTION')
FILE_PATH = os.getenv('FILE_PATH')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PROMPT_TEMPLATE = """
You are an expert in CadQuery and Python. Given the following context, generate **only** valid, functional CadQuery code that follows best practices. Ensure the code does not contain syntax errors and is executable.

Context:
{context}

---

Task: Generate CadQuery code for the following request:
{question}

**Guidelines:**
- Use `cq.Workplane` properly.
- Ensure all operations are valid and logically ordered.
- Include necessary imports (`import cadquery as cq`).
- Do not include explanations, only return valid Python code.
- If unsure, return the best possible attempt.

Output only the CadQuery code:
"""

# Ensure USER_AGENT is set
if 'USER_AGENT' not in os.environ:
    os.environ['USER_AGENT'] = "cadgpt"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

# Setup logging
logging.basicConfig(level=logging.INFO)

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY

def main():
    query_text = """
    make a hexagonal nut with 1/4 inch diameter and a thread of 1/8 inch
    """
    query_rag(query_text)

def query_rag(query_text: str):
    try:
        # Load Vector Store from Local ChromaDB
        persistent_client = chromadb.PersistentClient()
        vector_store = Chroma(
            client=persistent_client,
            collection_name=CHROMA_COLLECTION,
            embedding_function=get_embedding_function()
        )
        # Search the DB
        results = vector_store.similarity_search_with_score(query_text, k=5)

        context_text = ""

        context_text += """
        ---
        Example 1:
        Request: Create a cylinder with a 1-inch diameter and 2-inch height.
        Output:
        import cadquery as cq
        result = cq.Workplane("XY").circle(0.5).extrude(2)
        ---

        Example 2:
        Request: Create a nut with a 1/2 inch diameter.
        Output:
        import cadquery as cq
        diameter = 0.5
        height = 0.25
        result = cq.Workplane("XY").circle(diameter / 2).extrude(height)
        .faces("<Z").workplane().hole(diameter / 4)
        ---

        """

        # Rerank the results by score
        results = sorted(results, key=lambda x: x[1], reverse=True)

        for chunk in results:
            context_text += "\n\n---\n\n" + chunk[0].page_content

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        # logging.info(prompt)

        # Send prompt to OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in CadQuery and Python."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Filter the response to ensure it contains only code
        code_response = response.choices[0].message.content.strip()


        # Print Model Response
        sources = [chunk[0].metadata.get("id", None) for chunk in results]
        formatted_response = f"\n\n\033[32mResponse: {code_response}\033[0m\n\nSources: {sources}]"
        logging.info(formatted_response)

        # Write formatted response into jupyter notebook file
        notebook_filename = "result.ipynb"
        code_response_py = code_response.replace("```python","").replace("```","").strip()
        with open(notebook_filename, "r") as f:
                    nb = nbf.read(f, as_version=4)
        new_code = code_response_py+"\ndisplay(result)"
        new_code_cell = nbf.v4.new_code_cell(new_code)
        if "id" in new_code_cell:
            del new_code_cell["id"]
        nb.cells.append(new_code_cell)
        with open(notebook_filename, "w") as f:
            nbf.write(nb, f)



    except Exception as e:
        logging.error(f"An error occurred: {e}")

# query_rag using Vertex AI
# def query_rag(query_text:str):
#     try:
#         # Load Vector Store from Local ChromaDB
#         persistent_client = chromadb.PersistentClient()
#         vector_store = Chroma(
#                 client=persistent_client,
#                 collection_name=CHROMA_COLLECTION,
#                 embedding_function=get_embedding_function()
#             )
#         # Search the DB
#         results = vector_store.similarity_search_with_score(query_text, k=5)

#         context_text = ""

#         context_text += """
#         ---
#         Example 1:
#         Request: Create a cylinder with a 1-inch diameter and 2-inch height.
#         Output:
#         import cadquery as cq
#         result = cq.Workplane("XY").circle(0.5).extrude(2)
#         ---

#         Example 2:
#         Request: Create a nut with a 1/2 inch diameter.
#         Output:
#         import cadquery as cq
#         diameter = 0.5
#         height = 0.25
#         result = cq.Workplane("XY").circle(diameter / 2).extrude(height)
#         .faces("<Z").workplane().hole(diameter / 4)
#         ---

#         If the user asks for a hexagonal nut, you can reuse the code from the second example and modify it to create a 
#         hexagon instead of a circle. Same with other shapes like squares, triangles, etc. You can also modify the dimensions
#         to match the user's request.
#         """

#         # Rerank the results by score
#         results = sorted(results, key=lambda x: x[1], reverse=True)

#         for chunk in results:
#             context_text += "\n\n---\n\n" + chunk[0].page_content

#         prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#         prompt = prompt_template.format(context=context_text, question=query_text)
#         # logging.info(prompt)

#         # Send prompt into model
#         model = ChatVertexAI(model="gemini-1.5-flash")
#         response_text = model.invoke(prompt)

#         # Filter the response to ensure it contains only code
#         code_response = response_text.content.strip()

#         # Print Model Response
#         sources = [chunk[0].metadata.get("id", None) for chunk in results]
#         formatted_response = f"\n\n\033[32mResponse: {code_response}\033[0m\n\nSources: {sources}\n\nUsage Metadata: {response_text.usage_metadata}"
#         logging.info(formatted_response)

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()