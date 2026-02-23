from groq import Groq
import os
from dotenv import load_dotenv
from config import MODEL_CODER, DATA_PATH, RESULT_PATH

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_code(hypothesis):
    prompt = f"""
You are an expert ML engineer.

Write clean Python code using pandas + scikit-learn.

Task:
{hypothesis}

Rules:
- Load dataset from {DATA_PATH}
- Train model
- Print accuracy
- Save results to {RESULT_PATH}
- Handle errors
- No explanations, only code
"""

    chat = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL_CODER
    )

    return chat.choices[0].message.content
