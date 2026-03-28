from google import genai
import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

client=genai.Client(api_key = st.secrets["GEMINI_API_KEY"])

#for list the available model names
# for m in client.models.list():
#     print(m.name)
    
def embed_chunks(chunks):
    embeddings = []

    for chunk in chunks:
        response = client.models.embed_content(
            model="gemini-embedding-001",   
            contents=chunk
        )
        embeddings.append(response.embeddings[0].values)

    return embeddings