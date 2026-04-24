"""
feature_engineer.py  —  Automated AI Scientist v3.0
─────────────────────────────────────────────────────
LLM-driven Feature Engineering Agent.

What it does:
  1. Sends column names, dtypes, sample values, and correlations to the LLM
  2. LLM returns a JSON list of new features to create
  3. Each feature is defined as a Python expression over existing columns
  4. We eval() each expression safely, add valid ones to the DataFrame
  5. Returns the enriched DataFrame + a manifest of what was created

Why this matters:
  Raw features rarely tell the full story. For insurance data:
    age × smoker           → risk_score
    bmi / age              → bmi_age_ratio
    charges / bmi          → cost_efficiency
  These interactions are what tree models exploit manually;
  giving them explicitly as features turbo-charges linear models.

Safety:
  eval() is sandboxed to only the DataFrame columns + numpy + pandas.
  Any expression that raises an exception is silently skipped.
"""

import pandas as pd
import numpy as np
import json
import re
from typing import Optional


def _build_column_summary(df: pd.DataFrame) -> str:
    """Build a compact column summary for the LLM prompt."""
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if df[col].dtype == "object":
            top_vals = df[col].value_counts().head(4).index.tolist()
            lines.append(f"  {col} (categorical): values={top_vals}")
        else:
            mn  = round(float(df[col].min()), 2)
            mx  = round(float(df[col].max()), 2)
            avg = round(float(df[col].mean()), 2)
            lines.append(f"  {col} (numeric): min={mn}, max={mx}, mean={avg}")
    return "\n".join(lines)


def _build_correlation_summary(df: pd.DataFrame) -> str:
    """Return top 5 correlated pairs with target (if present)."""
    num_df = df.select_dtypes(include=[np.number])
    if "target" not in num_df.columns or len(num_df.columns) < 2:
        return ""
    corr = num_df.corr()["target"].drop("target").abs().sort_values(ascending=False)
    top5 = corr.head(5)
    lines = [f"  {col}: r={round(v,3)}" for col, v in top5.items()]
    return "Top feature-target correlations:\n" + "\n".join(lines)


def _safe_eval_feature(df: pd.DataFrame, expr: str) -> Optional[pd.Series]:
    """
    Safely evaluate a feature expression.
    Namespace: all columns as variables + np + pd.
    Returns None if expression raises any error.
    """
    namespace = {col: df[col] for col in df.columns}
    namespace["np"] = np
    namespace["pd"] = pd
    namespace["log"] = np.log1p   # safe log (handles 0)
    namespace["abs"] = np.abs
    namespace["sqrt"] = np.sqrt

    try:
        result = eval(expr, {"__builtins__": {}}, namespace)
        if isinstance(result, pd.Series):
            result = result.replace([np.inf, -np.inf], np.nan)
            result = result.fillna(result.median())
            return result
        return None
    except Exception:
        return None


def _call_llm_for_features(col_summary: str, corr_summary: str,
                             task: str, user_prompt: str) -> list:
    """
    Ask the LLM to suggest new features as Python expressions.
    Returns a list of dicts: [{name, expression, rationale}, ...]
    """
    from core.researcher import _call_groq

    prompt = f"""
You are an expert feature engineer. Analyze this dataset and suggest new features
that will improve {task} model performance.

Column summary:
{col_summary}

{corr_summary}

User goal: {user_prompt}

Suggest 5-8 new features. For each feature:
- name: short snake_case name
- expression: valid Python expression using column names as variables
  (available: all column names listed above, np.log1p(), np.abs(), np.sqrt())
- rationale: one sentence explaining why this feature helps

Rules:
- Only use column names that exist in the summary above
- For categorical columns encoded as numbers, treat them as numeric
- Use log1p() instead of log() to handle zeros safely
- Avoid division by zero — use (col + 1) when dividing
- Do NOT include the target column in any expression
- Keep expressions simple: multiplications, ratios, differences, log transforms

Return ONLY a valid JSON array. No explanation outside the JSON.
Example format:
[
  {{"name": "bmi_age_ratio", "expression": "bmi / (age + 1)", "rationale": "Captures age-adjusted BMI risk"}},
  {{"name": "log_charges", "expression": "log1p(charges)", "rationale": "Normalizes right-skewed cost distribution"}}
]
"""
    raw = _call_groq([{"role": "user", "content": prompt}], max_tokens=800)

    # Extract JSON array from response
    try:
        # Find JSON array in response
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return []


def run_feature_engineering(
    df: pd.DataFrame,
    task: str,
    user_prompt: str,
    max_features: int = 8,
) -> dict:
    """
    Main entry point. Returns enriched DataFrame + manifest.

    Parameters
    ----------
    df          : original DataFrame (with target column if supervised)
    task        : "classification" / "regression" / "unsupervised"
    user_prompt : user's original instruction (for LLM context)
    max_features: max new features to add

    Returns
    -------
    {
      "df_enriched"  : pd.DataFrame with new columns added,
      "new_features" : list of {name, expression, rationale, status},
      "n_added"      : int,
      "error"        : str (only if completely failed)
    }
    """
    try:
        # Work on copy without target
        feature_df = df.drop("target", axis=1, errors="ignore").copy()

        col_summary  = _build_column_summary(feature_df)
        corr_summary = _build_correlation_summary(df)

        # Ask LLM for feature suggestions
        suggestions  = _call_llm_for_features(
            col_summary, corr_summary, task, user_prompt
        )

        new_features = []
        df_enriched  = df.copy()

        for sug in suggestions[:max_features]:
            name  = str(sug.get("name", "")).strip()
            expr  = str(sug.get("expression", "")).strip()
            ratio = str(sug.get("rationale", "")).strip()

            if not name or not expr:
                continue

            # Sanitize name
            name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            if name in df_enriched.columns:
                name = f"{name}_eng"

            result = _safe_eval_feature(df_enriched, expr)
            if result is not None:
                df_enriched[name] = result
                new_features.append({
                    "name":       name,
                    "expression": expr,
                    "rationale":  ratio,
                    "status":     "added",
                })
            else:
                new_features.append({
                    "name":       name,
                    "expression": expr,
                    "rationale":  ratio,
                    "status":     "failed (invalid expression)",
                })

        n_added = sum(1 for f in new_features if f["status"] == "added")

        return {
            "df_enriched":  df_enriched,
            "new_features": new_features,
            "n_added":      n_added,
        }

    except Exception as e:
        return {
            "df_enriched":  df,
            "new_features": [],
            "n_added":      0,
            "error":        str(e),
        }
