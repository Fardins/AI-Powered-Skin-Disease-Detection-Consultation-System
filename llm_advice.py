from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Get API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize model
llm = ChatGroq(
    groq_api_key=groq_api_key,
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

