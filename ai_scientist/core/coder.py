from groq import Groq
import os
from dotenv import load_dotenv
from config import MODEL_CODER, DATA_PATH, RESULT_PATH

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_code(hypothesis):
    prompt = f"""
You are an expert ML engineer.

Task:
{hypothesis}

Write Python code using pandas and scikit-learn.

Requirements:
1. Load dataset from {DATA_PATH}
2. Train at least 3 ML models based on task
3. Calculate accuracy for each model
4. Print accuracy clearly
5. Save results to {RESULT_PATH} exactly like:

{{
 "models": [
   {{"name":"Logistic Regression","accuracy":0.85}},
   {{"name":"Random Forest","accuracy":0.91}},
   {{"name":"SVM","accuracy":0.88}}
 ]
}}

6. Handle missing target column safely
7. No markdown
8. Only pure python code
"""

    chat = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL_CODER
    )

    return chat.choices[0].message.content
