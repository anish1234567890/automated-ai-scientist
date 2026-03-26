from groq import Groq
import os
import time
from dotenv import load_dotenv

load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    timeout=60.0,
    max_retries=3,
)

MODEL = "llama-3.3-70b-versatile"


def _call_groq(messages: list, max_tokens: int = 512) -> str:
    """Wrapper with retry logic around every Groq call."""
    last_err = None
    for attempt in range(3):
        try:
            chat = client.chat.completions.create(
                messages=messages,
                model=MODEL,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return chat.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Groq API failed after 3 attempts: {last_err}")


# ── SUPERVISED ────────────────────────────────────────────────────

def decide_models(user_prompt: str) -> list:
    """LLM reads the user's natural language instruction and returns model names."""
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

Return ONLY a comma-separated list of model names exactly as shown. No explanation. No numbering.

Example output:
Random Forest, XGBoost, LightGBM, Gradient Boosting
"""
    text = _call_groq([{"role": "user", "content": prompt}], max_tokens=200)
    return [m.strip() for m in text.split(",") if m.strip()]


def generate_insight(results: dict, user_prompt: str) -> str:
    """LLM analyzes supervised experiment results and generates scientific insight."""
    task   = results.get("task", "unknown")
    metric = "Accuracy" if task == "classification" else "RMSE"

    model_summary = ""
    for m in results.get("models", []):
        score     = m.get("score")
        score_str = f"{score:.4f}" if score is not None else "Failed"
        params    = m.get("best_params", {})
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
    return _call_groq([{"role": "user", "content": prompt}], max_tokens=600)


# ── UNSUPERVISED ──────────────────────────────────────────────────

def decide_unsupervised_algos(user_prompt: str) -> list:
    """LLM reads the user's instruction and returns unsupervised algorithm names."""
    prompt = f"""
You are an expert ML scientist helping choose unsupervised learning algorithms.

User instruction:
{user_prompt}

Choose algorithms ONLY from this exact list:
- K-Means
- DBSCAN
- Agglomerative
- Gaussian Mixture
- Isolation Forest
- Local Outlier Factor

Mappings you MUST know:
- "kmeans" or "k means"               → K-Means
- "density" or "dbscan"               → DBSCAN
- "hierarchical" or "agglomerative"   → Agglomerative
- "gmm" or "gaussian"                 → Gaussian Mixture
- "anomaly" or "outlier detection"    → Isolation Forest
- "lof" or "local outlier"            → Local Outlier Factor

Rules:
- If user says "cluster" or "group" or "segment"  → K-Means, DBSCAN, Agglomerative, Gaussian Mixture
- If user says "anomaly" or "outlier"              → Isolation Forest, Local Outlier Factor
- If user says "all" or "try everything"           → all 6 algorithms
- If user explicitly names algorithms              → use exactly those
- If unclear                                       → K-Means, DBSCAN, Agglomerative

Return ONLY a comma-separated list. No explanation. No extra text.

Example output:
K-Means, DBSCAN, Agglomerative
"""
    return [m.strip() for m in _call_groq(
        [{"role": "user", "content": prompt}], max_tokens=100
    ).split(",") if m.strip()]


def generate_unsupervised_insight(results: dict, user_prompt: str) -> str:
    """LLM generates scientific analysis of clustering results."""
    clustering = results.get("clustering", [])

    summary = ""
    for c in clustering:
        sil = c.get("silhouette")
        dbi = c.get("davies_bouldin")
        ch  = c.get("calinski_harabasz")
        n_c = c.get("n_clusters_found", "?")
        if sil is not None:
            summary += (f"- {c['name']}: Silhouette={sil:.4f}, "
                        f"Davies-Bouldin={dbi}, Calinski-Harabasz={ch}, "
                        f"Clusters found={n_c}, "
                        f"Best params={c.get('best_params', {})}\n")
        else:
            summary += f"- {c['name']}: Failed — {c.get('error', 'unknown')}\n"

    pca_var = results.get("pca_variance", [])
    prompt  = f"""
You are a senior ML research scientist reviewing unsupervised learning results.

User goal: {user_prompt}
PCA explained variance (first components): {pca_var}

Clustering results:
{summary}

Write a 5-6 sentence scientific analysis covering:
1. Which algorithm found the best cluster structure and why (based on silhouette score)
2. What the optimal hyperparameters (e.g. n_clusters, eps) suggest about the data distribution
3. What the PCA variance explains about the data's dimensionality
4. One actionable recommendation: should the user try more clusters, different preprocessing, or a different algorithm?
5. Whether DBSCAN found meaningful density-based groups or if the data is globular (favoring K-Means/GMM)

Be specific and technical. Write as the AI Scientist in first person. Flowing paragraphs, no bullet points.
"""
    return _call_groq([{"role": "user", "content": prompt}], max_tokens=600)