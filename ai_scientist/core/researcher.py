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

def decide_models(user_prompt: str, df=None) -> list:
    """LLM reads the user's natural language instruction and returns model names."""
    dataset_info = ""
    if df is not None:
        try:
            target_type = ""
            if "target" in df.columns:
                target_type = "classification" if df["target"].nunique() < 15 else "regression"
            dataset_info = f"""
Dataset Summary:
Rows: {len(df)}
Columns: {df.shape[1]}
Missing: {df.isnull().sum().to_dict()}
Target type guess: {target_type}
"""
        except Exception:
            dataset_info = ""

    prompt = f"""
You are an expert ML scientist helping choose models for an AutoML experiment.

{dataset_info}

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

# ── V3 ADDITIONS ──────────────────────────────────────────────────

def generate_hypothesis(user_prompt: str, selected_models: list,
                        df=None, health: dict = None) -> str:
    """Generate a scientific hypothesis BEFORE training starts."""
    dataset_ctx = ""
    if df is not None:
        try:
            n, c = df.shape
            num  = df.select_dtypes("number").shape[1]
            cat  = df.select_dtypes("object").shape[1]
            miss = round(df.isnull().sum().sum() / max(n*c, 1) * 100, 1)
            ttype = ""
            if "target" in df.columns:
                ttype = "classification" if df["target"].nunique() < 15 else "regression"
            dataset_ctx = (f"Dataset: {n} rows, {c} cols, "
                           f"{num} numeric, {cat} categorical, "
                           f"{miss}% missing, task={ttype}")
        except Exception:
            pass

    health_ctx = ""
    if health:
        high = [i for i in health.get("issues", []) if i["severity"] == "high"]
        if high:
            health_ctx = "Data warnings: " + "; ".join(i["message"][:60] for i in high[:2])

    prompt = f"""
You are an expert ML scientist. Before any training, write a scientific hypothesis.

{dataset_ctx}
{health_ctx}
Models being tested: {', '.join(selected_models)}
User goal: {user_prompt}

Write 3-4 sentences starting with "My hypothesis is that..."
Predict which model will win and why. Be specific and technical.
"""
    return _call_groq([{"role": "user", "content": prompt}], max_tokens=350)


def generate_insight(results: dict, user_prompt: str,
                     hypothesis: str = "", shap_result: dict = None) -> str:
    """LLM insight — accepts optional hypothesis and SHAP for richer analysis."""
    task   = results.get("task", "unknown")
    metric = "Accuracy" if task == "classification" else "RMSE"

    model_summary = ""
    for m in results.get("models", []):
        score = m.get("score")
        s_str = f"{score:.4f}" if score is not None else "Failed"
        model_summary += f"- {m['name']}: {metric}={s_str} | params={m.get('best_params',{})}\n"

    ens = results.get("ensemble", {})
    ens_line = ""
    if ens and not ens.get("error"):
        ens_line = f"\nEnsemble (Top-3): {metric}={ens.get('cv_score','N/A')}"

    hyp_ctx = f"\nPre-experiment hypothesis: {hypothesis}" if hypothesis else ""

    shap_ctx = ""
    if shap_result and not shap_result.get("error"):
        top = shap_result.get("top_features", [])[:5]
        if top:
            shap_ctx = "\nSHAP top features: " + ", ".join(
                f"{t['feature']}({t['importance']:.4f})" for t in top)

    prompt = f"""
You are a senior ML research scientist reviewing AutoML experiment results.

User goal: {user_prompt}
Task: {task} | Metric: {metric} (5-fold CV)
{hyp_ctx}
Results:
{model_summary}{ens_line}
{shap_ctx}

Write a 6-7 sentence scientific analysis:
1. Was the hypothesis correct?
2. Which model performed best and why
3. What SHAP features reveal about the problem (if available)
4. What best hyperparameters suggest about the data
5. Did the ensemble improve over individual models?
6. One specific actionable recommendation

First person, flowing paragraphs, no bullet points.
"""
    return _call_groq([{"role": "user", "content": prompt}], max_tokens=700)


def generate_unsupervised_insight(results: dict, user_prompt: str,
                                   cluster_narrative: str = "") -> str:
    """Unsupervised insight — accepts cluster narrative for richer analysis."""
    clustering = results.get("clustering", [])
    summary = ""
    for c in clustering:
        sil = c.get("silhouette")
        if sil is not None:
            summary += (f"- {c['name']}: Silhouette={sil:.4f}, "
                        f"DB={c.get('davies_bouldin')}, CH={c.get('calinski_harabasz')}, "
                        f"Clusters={c.get('n_clusters_found','?')}, "
                        f"params={c.get('best_params',{})}\n")
        else:
            summary += f"- {c['name']}: Failed — {c.get('error','unknown')}\n"

    pca_var = results.get("pca_variance", [])
    profile_ctx = f"\nCluster profiles:\n{cluster_narrative}" if cluster_narrative else ""

    prompt = f"""
You are a senior ML research scientist reviewing unsupervised learning results.

User goal: {user_prompt}
PCA variance: {pca_var}

Results:
{summary}
{profile_ctx}

Write a 6-7 sentence analysis:
1. Which algorithm found best structure and why
2. What cluster profiles reveal about real-world meaning
3. What hyperparameters suggest about data distribution
4. What PCA variance explains about dimensionality
5. One actionable recommendation
6. Whether DBSCAN found density-based groups or data is globular

First person, flowing paragraphs, no bullet points.
"""
    return _call_groq([{"role": "user", "content": prompt}], max_tokens=700)
