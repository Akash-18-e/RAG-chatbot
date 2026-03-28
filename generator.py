from google import genai
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

client = genai.Client(api_key = st.secrets["GEMINI_API_KEY"])

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful assistant.

Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",   
        contents=prompt
    )

    return response.text