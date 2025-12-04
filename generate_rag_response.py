
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from data_store import data_retrieve
import os

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

client=genai.Client(api_key=api_key)

def generate_rag_response(query_text):
    """RAG - Send query + context to Gemini"""
    context = data_retrieve(query_text)
    rag_prompt = f"""You are a helpful banking assistant for DCB Bank. Use ONLY the following context to answer accurately.

        CONTEXT:
        {context}

        USER QUERY: {query_text}

        Answer concisely and accurately based on the context above."""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[rag_prompt]
    )
    print(response.text)
    return response.text.strip()