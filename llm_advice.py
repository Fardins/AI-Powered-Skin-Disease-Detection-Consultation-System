import os
import streamlit as st
from langchain_groq import ChatGroq

def get_groq_api_key():
    # 1. Try Streamlit secrets first
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    
    # 2. Fall back to environment variable
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    
    # 3. Key not found anywhere
    st.error("GROQ_API_KEY not found! Please set it in Streamlit secrets or .env file.")
    st.stop()

api_key = get_groq_api_key()

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=512
)


# Initialize model
api_key = get_groq_api_key()

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=512
)

def generate_advice(disease_name):
    prompt = f"""
You are a dermatologist AI.
Disease: {disease_name}

Format:
Explanation:
Treatment:
Next Steps:
Daily Tips:
"""
    result = llm.invoke(prompt)
    return result.content