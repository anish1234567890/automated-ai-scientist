"""
shap_explainer.py
─────────────────
SHAP (SHapley Additive exPlanations) feature importance for the
Automated AI Scientist.

What SHAP does:
  - For every prediction, computes how much each feature contributed
  - Based on cooperative game theory (Shapley values)
  - Model-agnostic: works with ANY sklearn estimator
  - Returns mean |SHAP| per feature = global feature importance

Fallback chain (most accurate → fastest):
  TreeExplainer  → tree models (RF, XGB, LightGBM, CatBoost, GB, ET)
  LinearExplainer → linear models (Ridge, Lasso, Logistic, etc.)
  KernelExplainer → everything else (SVM, KNN) — uses a sample of 100 rows
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ── Which model names use which explainer ────────────────────────
_TREE_MODELS = {
    "random forest", "xgboost", "lightgbm", "catboost",
    "gradient boosting", "extra trees", "decision tree",
    "adaboost", "bagging",
}
_LINEAR_MODELS = {
    "logistic regression", "linear regression", "ridge regression",
    "lasso regression", "elastic net", "sgd", "bayesian ridge",
    "huber", "lda",
}


def _explainer_type(model_name: str) -> str:
    n = model_name.lower().strip()
    for t in _TREE_MODELS:
        if t in n: return "tree"
    for l in _LINEAR_MODELS:
        if l in n: return "linear"
    return "kernel"


def compute_shap(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    model_name: str,
    task: str,
    max_samples: int = 200,
) -> dict:
    """
    Compute SHAP values for a fitted model.

    Parameters
    ----------
    model        : fitted sklearn estimator
    X_train      : training data (numpy array)
    X_test       : test data (numpy array)
    feature_names: list of column names
    model_name   : name string for explainer routing
    task         : "classification" or "regression"
    max_samples  : max rows to use for KernelExplainer (speed)

    Returns
    -------
    dict with keys:
      "feature_names"  : list[str]
      "mean_abs_shap"  : list[float]  — mean |SHAP| per feature
      "top_features"   : list[dict]   — top 10 [{feature, importance, rank}]
      "explainer_type" : str
      "error"          : str (only if failed)
    """
    if not SHAP_AVAILABLE:
        return {"error": "shap not installed. Run: pip install shap"}

    etype = _explainer_type(model_name)

    try:
        # Use test set for explanations (max max_samples rows for speed)
        X_explain = X_test[:max_samples] if len(X_test) > max_samples else X_test
        bg_data   = shap.maskers.Independent(X_train, max_samples=min(50, len(X_train)))

        if etype == "tree":
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)

            # For classifiers shap_values can be list-of-arrays (one per class)
            if isinstance(shap_values, list):
                # Multi-class: average over classes
                shap_arr = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_arr = np.abs(shap_values)

        elif etype == "linear":
            explainer   = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_explain)
            shap_arr    = np.abs(shap_values)

        else:  # kernel — slow but universal
            bg_sample   = shap.sample(X_train, min(50, len(X_train)))
            if task == "classification" and hasattr(model, "predict_proba"):
                fn = model.predict_proba
            else:
                fn = model.predict
            explainer   = shap.KernelExplainer(fn, bg_sample)
            shap_values = explainer.shap_values(
                X_explain[:50], silent=True
            )
            if isinstance(shap_values, list):
                shap_arr = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_arr = np.abs(shap_values)

        # Mean absolute SHAP per feature
        mean_shap = shap_arr.mean(axis=0).tolist()

        # Build sorted top-10
        pairs = sorted(
            zip(feature_names, mean_shap),
            key=lambda x: x[1],
            reverse=True,
        )
        top10 = [
            {"feature": f, "importance": round(v, 6), "rank": i + 1}
            for i, (f, v) in enumerate(pairs[:10])
        ]

        return {
            "feature_names":  feature_names,
            "mean_abs_shap":  [round(v, 6) for v in mean_shap],
            "top_features":   top10,
            "explainer_type": etype,
        }

    except Exception as e:
        return {"error": str(e)}


def run_shap_for_best_model(
    results: dict,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
) -> dict:
    """
    Re-trains the best model on full data and computes SHAP.
    Called after run_automl() returns.

    Returns shap_result dict (added to results["shap"]).
    """
    if not SHAP_AVAILABLE:
        return {"error": "shap not installed. Run: pip install shap"}

    from sklearn.model_selection import train_test_split
    from core.automl_engine import _build_model, _resolve_model_name
    from optuna.trial import FixedTrial

    valid = [m for m in results.get("models", []) if m.get("score") is not None]
    if not valid:
        return {"error": "No valid models to explain"}

    best_m   = valid[0]
    name     = _resolve_model_name(best_m["name"])
    params   = best_m.get("best_params", {})

    try:
        model = _build_model(FixedTrial(params), name, task)
        if model is None:
            return {"error": f"Could not build model: {best_m['name']}"}

        X_arr = X.values
        y_arr = y.values
        X_train, X_test, _, _ = train_test_split(
            X_arr, y_arr, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_arr[:len(X_train)])   # fit on train portion

        shap_result = compute_shap(
            model=model,
            X_train=X_train,
            X_test=X_test,
            feature_names=list(X.columns),
            model_name=best_m["name"],
            task=task,
        )
        shap_result["model_name"] = best_m["name"]
        return shap_result

    except Exception as e:
        return {"error": str(e)}
