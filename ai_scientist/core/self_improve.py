"""
self_improve.py  —  Automated AI Scientist v3.0
─────────────────────────────────────────────────
Iterative Self-Improvement Loop.

What it does:
  After the first experiment completes, this agent:
  1. Reads the full results (scores, failed models, SHAP features, health issues)
  2. Sends everything to the LLM and asks: "What specifically should change?"
  3. LLM returns a structured improvement plan as JSON
  4. System AUTOMATICALLY runs a second experiment with those changes
  5. Compares round 1 vs round 2 and reports the delta

This is the key difference between an AutoML tool and an AI Scientist.
A tool runs once. A scientist reflects, learns, and iterates.

Improvement types the LLM can suggest:
  - add_models      : "Try CatBoost — it handles mixed types better here"
  - remove_models   : "Drop KNN — too slow for 50k rows"
  - increase_trials : "XGBoost needs more tuning — increase trials to 50"
  - focus_features  : "SHAP shows only 3 features matter — drop the rest"
  - change_strategy : "Switch to regression — target has 45 unique values"
  - fix_data        : "Address class imbalance before retraining"
"""

import json
import re
from typing import Optional


def _call_llm_for_improvement_plan(
    results: dict,
    health:  dict,
    shap_result: dict,
    user_prompt: str,
    round_num: int,
) -> dict:
    """
    Ask the LLM to analyze results and return a structured improvement plan.
    Returns a dict with specific, actionable changes.
    """
    from core.researcher import _call_groq

    task   = results.get("task", "unknown")
    metric = "Accuracy" if task == "classification" else "RMSE"

    # Build model summary
    model_lines = []
    for m in results.get("models", []):
        score = m.get("score")
        if score is not None:
            model_lines.append(f"  {m['name']:25s}: {metric}={score:.4f}")
        else:
            model_lines.append(f"  {m['name']:25s}: FAILED — {m.get('error','?')}")
    model_summary = "\n".join(model_lines)

    # Best score
    valid  = [m for m in results.get("models", []) if m.get("score") is not None]
    best   = valid[0] if valid else {}
    best_s = f"{best.get('score', 0):.4f}" if best else "N/A"

    # SHAP context
    shap_line = ""
    if shap_result and not shap_result.get("error"):
        top3 = shap_result.get("top_features", [])[:3]
        shap_line = "Top SHAP features: " + ", ".join(
            f"{t['feature']}(imp={t['importance']:.4f})" for t in top3
        )

    # Health context
    health_issues = ""
    if health and health.get("issues"):
        high = [i for i in health["issues"] if i["severity"] == "high"]
        if high:
            health_issues = "Data quality issues: " + "; ".join(
                i["message"][:80] for i in high[:2]
            )

    # Ensemble context
    ens = results.get("ensemble", {})
    ens_line = ""
    if ens and not ens.get("error"):
        ens_line = f"Ensemble CV {metric}: {ens.get('cv_score','N/A')}"

    prompt = f"""
You are an expert AutoML scientist conducting Round {round_num} of iterative experiment improvement.

EXPERIMENT RESULTS (Round {round_num - 1}):
Task: {task}
Best model: {best.get('name','?')} — {metric}: {best_s}
{ens_line}

All model scores:
{model_summary}

{shap_line}
{health_issues}

User's original goal: {user_prompt}

Analyze these results and create a specific improvement plan for Round {round_num}.

Return ONLY a valid JSON object with this exact structure:
{{
  "reasoning": "2-3 sentence explanation of what you observed and why these changes will help",
  "add_models": ["model names to add from: Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, Extra Trees, SVM, KNN, Ridge Regression, Lasso Regression"],
  "remove_models": ["model names to remove — slow or underperforming ones"],
  "increase_trials": true or false,
  "new_n_trials": 35,
  "focus_on_top_features": true or false,
  "prompt_override": "revised experiment instruction that incorporates the improvement strategy",
  "expected_improvement": "one sentence on what metric improvement you expect and why"
}}

Rules:
- Only add models appropriate for the task type ({task})
- Remove at most 2 models
- Only increase trials if best model is tree-based and margin of improvement seems possible
- If SHAP shows < 4 features matter, set focus_on_top_features to true
- prompt_override MUST be a complete instruction, not empty
- Be specific and justified — vague suggestions are useless

Return ONLY the JSON. No text before or after.
"""
    raw = _call_groq([{"role": "user", "content": prompt}], max_tokens=600)

    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    # Fallback: minimal plan
    return {
        "reasoning":              "Insufficient data to form improvement plan.",
        "add_models":             [],
        "remove_models":          [],
        "increase_trials":        False,
        "new_n_trials":           25,
        "focus_on_top_features":  False,
        "prompt_override":        user_prompt,
        "expected_improvement":   "Unknown",
    }


def build_improved_experiment(
    plan:             dict,
    original_prompt:  str,
    selected_models:  list,
    n_trials:         int,
    shap_result:      dict,
) -> dict:
    """
    Apply the improvement plan to produce the next experiment configuration.

    Returns
    -------
    {
      "prompt"          : str   — possibly revised prompt
      "selected_models" : list  — adjusted model list
      "n_trials"        : int   — possibly increased
      "drop_columns"    : list  — columns to drop if focusing on SHAP
      "changes_applied" : list  — human-readable list of changes
    }
    """
    changes   = []
    models    = list(selected_models)

    # Add suggested models
    for m in plan.get("add_models", []):
        if m not in models:
            models.append(m)
            changes.append(f"➕ Added model: {m}")

    # Remove underperforming models
    for m in plan.get("remove_models", []):
        if m in models and len(models) > 1:
            models.remove(m)
            changes.append(f"➖ Removed model: {m} (underperforming)")

    # Increase trials
    new_trials = n_trials
    if plan.get("increase_trials") and plan.get("new_n_trials"):
        new_trials = max(n_trials, int(plan["new_n_trials"]))
        if new_trials > n_trials:
            changes.append(f"⬆️ Increased Optuna trials: {n_trials} → {new_trials}")

    # Focus on top SHAP features
    drop_columns = []
    if plan.get("focus_on_top_features") and shap_result and not shap_result.get("error"):
        top_feats = [t["feature"] for t in shap_result.get("top_features", [])[:5]]
        if top_feats:
            drop_columns = []   # will be resolved in app.py using full feature list
            changes.append(f"🎯 Focusing on top SHAP features: {', '.join(top_feats)}")

    # Revised prompt
    new_prompt = plan.get("prompt_override", original_prompt) or original_prompt
    if new_prompt != original_prompt:
        changes.append(f"✏️ Revised prompt based on results analysis")

    return {
        "prompt":           new_prompt,
        "selected_models":  models,
        "n_trials":         new_trials,
        "drop_columns":     drop_columns,
        "changes_applied":  changes,
        "reasoning":        plan.get("reasoning", ""),
        "expected_improvement": plan.get("expected_improvement", ""),
    }


def compare_rounds(round1: dict, round2: dict) -> dict:
    """
    Compare two experiment results and compute improvement deltas.

    Returns
    -------
    {
      "task"           : str,
      "metric"         : str,
      "round1_best"    : {name, score},
      "round2_best"    : {name, score},
      "delta"          : float,
      "improved"       : bool,
      "pct_change"     : float,
      "winner_round"   : int,
      "model_changes"  : list of str,
    }
    """
    task   = round1.get("task", round2.get("task", "unknown"))
    metric = "Accuracy" if task == "classification" else "RMSE"

    def get_best(results):
        valid = [m for m in results.get("models", []) if m.get("score") is not None]
        if not valid:
            return {"name": "None", "score": None}
        if task == "classification":
            return max(valid, key=lambda x: x["score"])
        else:
            return min(valid, key=lambda x: x["score"])

    b1 = get_best(round1)
    b2 = get_best(round2)

    s1 = b1.get("score") or 0
    s2 = b2.get("score") or 0

    if task == "classification":
        delta    = round(s2 - s1, 6)
        improved = delta > 0
    else:
        delta    = round(s1 - s2, 6)   # positive = RMSE dropped = improved
        improved = delta > 0

    pct_change = round(abs(delta) / max(abs(s1), 1e-9) * 100, 2)
    winner     = 2 if improved else 1

    # Model changes
    m1_names = {m["name"] for m in round1.get("models", []) if m.get("score") is not None}
    m2_names = {m["name"] for m in round2.get("models", []) if m.get("score") is not None}
    added    = m2_names - m1_names
    removed  = m1_names - m2_names
    changes  = [f"Added: {', '.join(added)}"] if added else []
    changes += [f"Removed: {', '.join(removed)}"] if removed else []

    return {
        "task":         task,
        "metric":       metric,
        "round1_best":  {"name": b1.get("name", "?"), "score": s1},
        "round2_best":  {"name": b2.get("name", "?"), "score": s2},
        "delta":        delta,
        "improved":     improved,
        "pct_change":   pct_change,
        "winner_round": winner,
        "model_changes": changes,
    }
