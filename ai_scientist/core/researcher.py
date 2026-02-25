from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"


def decide_models(user_prompt: str) -> list:
    """
    LLM reads the user's natural language instruction
    and returns a list of model names to try.
    """

    prompt = f"""
You are an expert ML scientist helping choose models for an AutoML experiment.

User instruction:
{user_prompt}

Choose models ONLY from this exact list:
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting
- AdaBoost
- Extra Trees
- Bagging
- Decision Tree
- KNN
- SVM
- Logistic Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- SGD
- Bayesian Ridge
- Huber
- Naive Bayes
- LDA

Mappings you MUST know:
- "l1 regression" or "lasso"       → Lasso Regression
- "l2 regression" or "ridge"       → Ridge Regression
- "elastic net" or "elasticnet"    → Elastic Net
- "gbm" or "gradient boosting"     → Gradient Boosting
- "ada" or "adaboost"              → AdaBoost
- "et" or "extra trees"            → Extra Trees
- "knn" or "k nearest neighbors"   → KNN
- "lgbm" or "lightgbm"             → LightGBM
- "cat" or "catboost"              → CatBoost
- "nb" or "naive bayes"            → Naive Bayes
- "dt" or "decision tree"          → Decision Tree
- "linear discriminant" or "lda"   → LDA
- "bayesian" or "bayesian ridge"   → Bayesian Ridge
- "huber regression"               → Huber
- "stochastic gradient" or "sgd"   → SGD

Task rules:
- Classification task → use ONLY: Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, AdaBoost, Extra Trees, Decision Tree, KNN, SVM, Logistic Regression, Naive Bayes, LDA, SGD
- Regression task → use ONLY: Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, AdaBoost, Extra Trees, Decision Tree, KNN, SVM, Linear Regression, Ridge Regression, Lasso Regression, Elastic Net, SGD, Bayesian Ridge, Huber
- If user says "all" or "best" → pick top 5 most powerful for the task
- If user explicitly names models → use exactly those (after mapping above)
- If task unclear → default to: Random Forest, XGBoost, LightGBM, Gradient Boosting, SVM

Return ONLY a comma-separated list of model names exactly as shown in the list above. No explanation. No numbering. No extra text.

Example output:
Random Forest, XGBoost, LightGBM, Gradient Boosting
"""

    chat = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL,
    )

    text = chat.choices[0].message.content.strip()
    # Clean up and parse
    models = [m.strip() for m in text.split(",") if m.strip()]
    return models


def generate_insight(results: dict, user_prompt: str) -> str:
    """
    LLM analyzes experiment results and generates
    a scientific explanation of what happened and why.
    """

    task = results.get("task", "unknown")
    metric = "Accuracy" if task == "classification" else "RMSE"

    model_summary = ""
    for m in results.get("models", []):
        score = m.get("score")
        score_str = f"{score:.4f}" if score is not None else "Failed"
        params = m.get("best_params", {})
        model_summary += f"- {m['name']}: {metric} = {score_str} | Best params: {params}\n"

    prompt = f"""
You are a senior ML research scientist reviewing AutoML experiment results.

User goal: {user_prompt}
Task type: {task}
Metric used: {metric}

Experiment results:
{model_summary}

Write a 5-6 sentence scientific analysis covering:
1. Which model performed best and a likely reason why
2. What the best hyperparameter values suggest about the data structure
3. One specific actionable recommendation to improve results further
4. Whether there are signs of overfitting or underfitting based on the scores

Be specific, technical, and insightful. Write as the AI Scientist agent in first person.
Do not use bullet points. Write in flowing paragraphs.
"""

    chat = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL,
    )

    return chat.choices[0].message.content.strip()