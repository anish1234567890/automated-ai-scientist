from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_hypothesis(dataset_info):

    prompt = f"""
You are an expert ML scientist.

Dataset info:
{dataset_info}

Based on user instruction, decide which ML models to try.

Examples:
- If user says deep learning → use neural networks
- If user says fast → use logistic regression
- If user says accuracy → use boosting

Return 3 model strategies clearly.
"""

    chat = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )

    return chat.choices[0].message.content
