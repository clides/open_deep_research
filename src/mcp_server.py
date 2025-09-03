import os
import json
import random
import numpy as np
from typing import List
from dotenv import load_dotenv

from fastmcp import FastMCP
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

app = FastMCP("Codebase Consulting Server")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

@app.tool()
def extract_relevant_file_content(query: str) -> str:
    """
    An MCP tool that extracts and concatenates content from relevant files
    that are useful for answering the user query.
    Returns a single string containing all relevant content.
    """
    random.seed(42)
    np.random.seed(42)

    prompt = f"""
    Extract concise keywords or relevant keywords (that may not be from the query but are relevant) from the following query that can be used to search relevant source files. DO NOT include any words that are not relevant to the query. Include only the MOST relevant keywords. Do not include generic keywords like "code", "file", "files", "project", "repository", etc. Only include keywords that are specific to the query.
    Your response MUST be a list of the keywords seperated by spaces. Do not include any other text, comments, or markdown.

    Query: "{query}"
    """
    response = model.generate_content(prompt).text.strip()
    keywords = response.split()

    print(f"DEBUG: Extracted keywords from query: {keywords}")

    skipped_dirs = {'.git', '__pycache__', 'node_modules', 'venv', '.env', '.vscode', '.idea'}
    skipped_exts = {'.pyc', '.so', '.dll', '.exe', '.jpg', '.png', '.gif', '.pdf', '.zip', '.tar'}

    relevant_files = []
    for root, _, files in os.walk("."):
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), ".")

            # Skip if file extension is in skipped extensions
            if any(rel_path.endswith(ext) for ext in skipped_exts):
                continue
            
            # Skip if any part of the path contains a skipped directory
            path_parts = rel_path.split(os.sep)
            if any(part in skipped_dirs for part in path_parts):
                continue

            try:
                with open(rel_path, "r", encoding="utf-8", errors="ignore") as file:
                    content = file.read()
                # Check if any keyword is in filename or content
                if any(kw.lower() in rel_path.lower() or kw.lower() in content.lower() for kw in keywords):
                    relevant_files.append((rel_path, content))
            except Exception as e:
                print(f"Error reading {rel_path}: {e}")


    documents = []
    for file_path, content in relevant_files:
        documents.append(Document(
            page_content=content,
            metadata={"source": file_path}
        ))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)

    # embeddings = OllamaEmbeddings(model='openhermes', base_url="http://localhost:11434")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={
            'normalize_embeddings': True, 
            'batch_size': 32,  # Larger batch size for GPU
            'show_progress_bar': True
        }
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    relevant_chunks = retriever.get_relevant_documents(query)

    result = []
    for i, chunk in enumerate(relevant_chunks):
        source = chunk.metadata.get('source', 'unknown')
        
        result.append(f"source: {source}")
        result.append(chunk.page_content)
        result.append("")

    if result and result[-1] == "":
        result.pop()

    return "\n".join(result)


if __name__ == "__main__":
    app.run(transport="http", host="127.0.0.1", port=8080)
