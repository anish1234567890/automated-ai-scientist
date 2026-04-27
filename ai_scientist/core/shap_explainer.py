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

    try:
        etype = _explainer_type(model_name)

        def _to_numpy(data):
            return np.array(data) if not isinstance(data, np.ndarray) else data

        def _normalize_shap_matrix(sv):
            if isinstance(sv, list):
                sv = np.array(sv[0]) if len(sv) > 0 else np.array([])
            else:
                sv = np.array(sv)

            if sv.ndim == 1:
                sv = sv.reshape(1, -1)
            elif sv.ndim == 3:
                sv = sv[:, :, 0]
            elif sv.ndim > 3:
                sv = sv.reshape(sv.shape[0], -1)
            return sv

        X_train = _to_numpy(X_train)
        X_test = _to_numpy(X_test)

        # Use test set for explanations (max max_samples rows for speed)
        X_explain = X_test[:max_samples] if len(X_test) > max_samples else X_test
        X_sample = _to_numpy(X_explain[:100])
        if X_sample.shape[0] == 0:
            return {"error": "No rows available for SHAP explanation"}

        if etype == "tree":
            explainer   = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample)
            sv = _normalize_shap_matrix(sv)

        elif etype == "linear":
            explainer   = shap.LinearExplainer(model, _to_numpy(X_train))
            sv = explainer.shap_values(X_sample)
            sv = _normalize_shap_matrix(sv)

        else:  # kernel — slow but universal
            bg_size = min(50, X_train.shape[0])
            bg_idx = np.random.choice(X_train.shape[0], bg_size, replace=False)
            background = np.array(X_train[bg_idx])
            if task == "classification" and hasattr(model, "predict_proba"):
                fn = model.predict_proba
            else:
                fn = model.predict
            explainer   = shap.KernelExplainer(fn, background)
            sv = explainer.shap_values(
                np.array(X_sample[:50]), silent=True
            )
            sv = _normalize_shap_matrix(sv)

        if sv.ndim != 2:
            sv = sv.reshape(sv.shape[0], -1) if sv.ndim > 1 else sv.reshape(1, -1)

        # Mean absolute SHAP per feature
        mean_shap = np.abs(sv).mean(axis=0).tolist()

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
    feature_names = list(X.columns) if hasattr(X, "columns") else None
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if hasattr(y, "values"):
        y = y.values
    y = np.array(y)

    if not SHAP_AVAILABLE:
        return {"error": "shap not installed. Run: pip install shap"}

    from sklearn.model_selection import train_test_split
    from core.automl_engine import _build_model, _resolve_model_name
    from optuna.trial import FixedTrial

    valid = [m for m in results.get("models", []) if m.get("score") is not None]
    if not valid:
        return {"error": "No valid models to explain"}

    best_m       = valid[0]
    raw_name     = str(best_m.get("name", ""))
    resolved_name = _resolve_model_name(raw_name)
    params       = best_m.get("best_params", {})

    try:
        model = _build_model(FixedTrial(params), resolved_name, task)
        if model is None:
            return {"error": f"Could not build model: {raw_name}"}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]

        shap_result = compute_shap(
            model=model,
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_names,
            model_name=resolved_name,
            task=task,
        )
        shap_result["model_name"] = raw_name
        shap_result["resolved_model_name"] = resolved_name
        shap_result["best_model_params"] = params
        return shap_result

    except Exception as e:
        return {"error": str(e)}
