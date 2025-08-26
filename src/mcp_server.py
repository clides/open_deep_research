import os
import json
from typing import List
from fastmcp import FastMCP
import google.generativeai as genai
from dotenv import load_dotenv

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
    prompt = f"""
    Extract concise keywords or relevant keywords (that may not be from the query but are relevant) from the following query that can be used to search relevant source files. DO NOT include any words that are not relevant to the query. Include only the MOST relevant keywords.
    Your response MUST be a list of the keywords seperated by spaces. Do not include any other text, comments, or markdown.

    Query: "{query}"
    """
    response = model.generate_content(prompt).text.strip()

    keywords = response.split()
    keywords = ["test_openrouter_integration"]

    print(f"DEBUG: Extracted keywords from query: {keywords}")

    relevant_information = ""
    for root, _, files in os.walk("."):
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), ".")
            try:
                with open(rel_path, "r", encoding="utf-8", errors="ignore") as file:
                    content = file.read()
                # Check if any keyword is in filename or content
                if any(kw.lower() in rel_path.lower() or kw.lower() in content.lower() for kw in keywords):
                    relevant_information += rel_path + "\n" + content + "\n\n"
            except Exception as e:
                print(f"Error reading {rel_path}: {e}")

    return relevant_information.strip()

if __name__ == "__main__":
    app.run(transport="http", host="127.0.0.1", port=8080)
