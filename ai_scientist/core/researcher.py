from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_hypothesis(dataset_info):

    prompt = f"""
You are a senior data scientist.

Dataset info:
{dataset_info}

Generate 3 ML hypotheses.
Return short bullet points only.
"""

    chat = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )

    return chat.choices[0].message.content
