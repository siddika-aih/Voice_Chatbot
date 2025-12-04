
# import numpy as np
# from google import genai
# from google.genai import types
# from dotenv import load_dotenv
# from data_store import data_retrieve
# import os

# load_dotenv()

# api_key=os.getenv("GEMINI_API_KEY")

# client=genai.Client(api_key=api_key)

# def generate_rag_response(query_text):
#     """RAG - Send query + context to Gemini"""
#     context = data_retrieve(query_text)
#     rag_prompt = f"""You are a helpful banking assistant for DCB Bank. Use ONLY the following context to answer accurately.

#         CONTEXT:
#         {context}

#         USER QUERY: {query_text}

#         Answer concisely and accurately based on the context above."""
    
#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=[rag_prompt]
#     )
#     return response.text.strip()


# enhanced_rag.py - Streaming LLM responses with context
import asyncio
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def stream_rag_response(query: str, context: str, conversation_history: list):
    """Stream RAG response with conversation context"""
    
    # Build conversation context
    history_text = ""
    if conversation_history:
        recent_history = conversation_history[-3:]  # Last 3 turns
        history_text = "\n".join([
            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            for turn in recent_history
        ])
    
    # Enhanced prompt with guardrails
    prompt = f"""You are a helpful and accurate voice assistant for DCB Bank. Follow these rules strictly:

1. Use ONLY the context provided below to answer
2. If information is not in context, say "I don't have that information. Please contact DCB Bank customer care at 1800..."
3. Be concise (2-3 sentences max for voice)
4. Sound natural and conversational
5. Never make up information

CONVERSATION HISTORY:
{history_text}

RELEVANT CONTEXT:
{context}

USER QUERY: {query}

RESPONSE (concise and accurate):"""
    
    # Stream response from Gemini
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=[prompt]
    )
    return response.text.strip()