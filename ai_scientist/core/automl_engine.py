import pandas as pd
import numpy as np
import optuna
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    BaggingClassifier, BaggingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor,
    BayesianRidge, HuberRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMClassifier = None
    LGBMRegressor  = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None
    CatBoostRegressor  = None


# ── Model registry ────────────────────────────────────────────────
# Maps lowercase name → (ClassifierClass, RegressorClass, task_support)
# task_support: "both", "classification", "regression"
MODEL_REGISTRY = {
    "random forest":          (RandomForestClassifier,        RandomForestRegressor,        "both"),
    "xgboost":                (XGBClassifier,                  XGBRegressor,                  "both"),
    "lightgbm":               (LGBMClassifier if LIGHTGBM_AVAILABLE else None,   LGBMRegressor if LIGHTGBM_AVAILABLE else None,   "both"),
    "catboost":               (CatBoostClassifier if CATBOOST_AVAILABLE else None, CatBoostRegressor if CATBOOST_AVAILABLE else None, "both"),
    "gradient boosting":      (GradientBoostingClassifier,    GradientBoostingRegressor,    "both"),
    "adaboost":               (AdaBoostClassifier,             AdaBoostRegressor,             "both"),
    "extra trees":            (ExtraTreesClassifier,           ExtraTreesRegressor,           "both"),
    "bagging":                (BaggingClassifier,              BaggingRegressor,              "both"),
    "decision tree":          (DecisionTreeClassifier,         DecisionTreeRegressor,         "both"),
    "knn":                    (KNeighborsClassifier,           KNeighborsRegressor,           "both"),
    "k-nearest neighbors":    (KNeighborsClassifier,           KNeighborsRegressor,           "both"),
    "svm":                    (SVC,                            SVR,                           "both"),
    "logistic regression":    (LogisticRegression,             None,                          "classification"),
    "linear regression":      (None,                           LinearRegression,              "regression"),
    "ridge regression":       (None,                           Ridge,                         "regression"),
    "lasso regression":       (None,                           Lasso,                         "regression"),
    "elastic net":            (None,                           ElasticNet,                    "regression"),
    "sgd":                    (SGDClassifier,                  SGDRegressor,                  "both"),
    "bayesian ridge":         (None,                           BayesianRidge,                 "regression"),
    "huber":                  (None,                           HuberRegressor,                "regression"),
    "naive bayes":            (GaussianNB,                     None,                          "classification"),
    "lda":                    (LinearDiscriminantAnalysis,     None,                          "classification"),
}


def detect_task(y: pd.Series) -> str:
    if y.dtype == "object" or y.nunique() < 15:
        return "classification"
    return "regression"


def preprocess(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    X = X.fillna(X.median(numeric_only=True))
    return X


def _resolve_model_name(user_name: str) -> str:
    """Fuzzy match user model name to registry key."""
    n = user_name.lower().strip()
    # exact match
    if n in MODEL_REGISTRY:
        return n
    # partial match
    for key in MODEL_REGISTRY:
        if key in n or n in key:
            return key
    return n


def _make_objective(model_name: str, task: str, X_train, X_test, y_train, y_test):
    name = model_name.lower().strip()

    def objective(trial):
        model = _build_model(trial, name, task)
        if model is None:
            return 0.0 if task == "classification" else 1e9
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if task == "classification":
            return accuracy_score(y_test, preds)
        else:
            return float(np.sqrt(mean_squared_error(y_test, preds)))

    return objective


def _build_model(trial, name: str, task: str):
    """Build model with Optuna-suggested hyperparameters."""

    is_cls = (task == "classification")

    if "random forest" in name:
        p = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
            "max_depth":         trial.suggest_int("max_depth", 3, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        return (RandomForestClassifier if is_cls else RandomForestRegressor)(**p, random_state=42, n_jobs=-1)

    elif "xgboost" in name:
        if not XGBOOST_AVAILABLE: return None
        p = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
        }
        return (XGBClassifier if is_cls else XGBRegressor)(**p, random_state=42, verbosity=0, n_jobs=-1)

    elif "lightgbm" in name or "lgbm" in name:
        if not LIGHTGBM_AVAILABLE: return None
        p = {
            "n_estimators":   trial.suggest_int("n_estimators", 50, 400),
            "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth":      trial.suggest_int("max_depth", 3, 12),
            "num_leaves":     trial.suggest_int("num_leaves", 20, 150),
            "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        return (LGBMClassifier if is_cls else LGBMRegressor)(**p, random_state=42, n_jobs=-1, verbose=-1)

    elif "catboost" in name:
        if not CATBOOST_AVAILABLE: return None
        p = {
            "iterations":   trial.suggest_int("iterations", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "depth":         trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
        return (CatBoostClassifier if is_cls else CatBoostRegressor)(**p, random_state=42, verbose=0)

    elif "gradient boosting" in name:
        p = {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth":     trial.suggest_int("max_depth", 2, 8),
            "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
            "max_features":  trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        return (GradientBoostingClassifier if is_cls else GradientBoostingRegressor)(**p, random_state=42)

    elif "adaboost" in name:
        p = {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0),
        }
        return (AdaBoostClassifier if is_cls else AdaBoostRegressor)(**p, random_state=42)

    elif "extra trees" in name:
        p = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
            "max_depth":         trial.suggest_int("max_depth", 3, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        return (ExtraTreesClassifier if is_cls else ExtraTreesRegressor)(**p, random_state=42, n_jobs=-1)

    elif "bagging" in name:
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 100),
            "max_samples":  trial.suggest_float("max_samples", 0.5, 1.0),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }
        return (BaggingClassifier if is_cls else BaggingRegressor)(**p, random_state=42, n_jobs=-1)

    elif "decision tree" in name:
        p = {
            "max_depth":         trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "criterion":         trial.suggest_categorical("criterion",
                                    ["gini", "entropy"] if is_cls else ["squared_error", "friedman_mse"]),
        }
        return (DecisionTreeClassifier if is_cls else DecisionTreeRegressor)(**p, random_state=42)

    elif "knn" in name or "k-nearest" in name or "k nearest" in name:
        p = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
            "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric":      trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
        }
        return (KNeighborsClassifier if is_cls else KNeighborsRegressor)(**p)

    elif "svm" in name:
        p = {
            "C":      trial.suggest_float("C", 0.01, 100.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
        }
        if is_cls:
            p["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            return SVC(**p)
        else:
            p["epsilon"] = trial.suggest_float("epsilon", 0.01, 1.0)
            return SVR(**p)

    elif "logistic" in name:
        if not is_cls: return None
        p = {
            "C":       trial.suggest_float("C", 0.01, 100.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
        }
        solver = "saga" if p["penalty"] in ["l1", "elasticnet"] else "lbfgs"
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if p["penalty"] == "elasticnet" else None
        return LogisticRegression(**p, solver=solver, l1_ratio=l1_ratio, max_iter=1000, random_state=42)

    elif name == "linear regression":
        return LinearRegression()

    elif "ridge" in name:
        alpha = trial.suggest_float("alpha", 0.01, 100.0, log=True)
        return Ridge(alpha=alpha)

    elif "lasso" in name:
        alpha = trial.suggest_float("alpha", 0.0001, 10.0, log=True)
        return Lasso(alpha=alpha, max_iter=5000)

    elif "elastic net" in name or "elasticnet" in name:
        p = {
            "alpha":    trial.suggest_float("alpha", 0.0001, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
        return ElasticNet(**p, max_iter=5000)

    elif "sgd" in name:
        p = {
            "alpha":        trial.suggest_float("alpha", 1e-5, 1.0, log=True),
            "learning_rate": trial.suggest_categorical("learning_rate", ["optimal", "invscaling", "adaptive"]),
        }
        if is_cls:
            p["loss"] = trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber"])
            return SGDClassifier(**p, random_state=42, max_iter=1000)
        else:
            p["loss"] = trial.suggest_categorical("loss", ["squared_error", "huber", "epsilon_insensitive"])
            return SGDRegressor(**p, random_state=42, max_iter=1000)

    elif "bayesian ridge" in name or "bayesian" in name:
        if is_cls: return None
        p = {
            "alpha_1": trial.suggest_float("alpha_1", 1e-7, 1e-4, log=True),
            "alpha_2": trial.suggest_float("alpha_2", 1e-7, 1e-4, log=True),
            "lambda_1": trial.suggest_float("lambda_1", 1e-7, 1e-4, log=True),
            "lambda_2": trial.suggest_float("lambda_2", 1e-7, 1e-4, log=True),
        }
        return BayesianRidge(**p)

    elif "huber" in name:
        if is_cls: return None
        p = {
            "epsilon": trial.suggest_float("epsilon", 1.01, 2.0),
            "alpha":   trial.suggest_float("alpha", 1e-5, 1.0, log=True),
        }
        return HuberRegressor(**p, max_iter=500)

    elif "naive bayes" in name or "gaussiannb" in name:
        if not is_cls: return None
        var_smoothing = trial.suggest_float("var_smoothing", 1e-11, 1e-7, log=True)
        return GaussianNB(var_smoothing=var_smoothing)

    elif "lda" in name or "linear discriminant" in name:
        if not is_cls: return None
        return LinearDiscriminantAnalysis()

    return None


def _generate_final_code(results: dict, data_path: str) -> str:
    task   = results.get("task", "classification")
    models = [m for m in results.get("models", []) if m.get("score") is not None]
    metric = "Accuracy" if task == "classification" else "RMSE"

    lines = []
    lines.append("# ================================================================")
    lines.append("# AUTO-GENERATED BY AUTOMATED AI SCIENTIST")
    lines.append(f"# Task  : {task.upper()}")
    lines.append(f"# Metric: {metric}")
    lines.append("# Hyperparameters tuned by Optuna TPE sampler")
    lines.append("# ================================================================")
    lines.append("")
    lines.append("import pandas as pd")
    lines.append("import numpy as np")
    lines.append("import warnings")
    lines.append("warnings.filterwarnings(\"ignore\")")
    lines.append("from sklearn.model_selection import train_test_split")
    lines.append("from sklearn.preprocessing import LabelEncoder")
    lines.append("from sklearn.metrics import accuracy_score, mean_squared_error")
    lines.append("from sklearn.ensemble import (")
    lines.append("    RandomForestClassifier, RandomForestRegressor,")
    lines.append("    GradientBoostingClassifier, GradientBoostingRegressor,")
    lines.append("    AdaBoostClassifier, AdaBoostRegressor,")
    lines.append("    ExtraTreesClassifier, ExtraTreesRegressor,")
    lines.append("    BaggingClassifier, BaggingRegressor")
    lines.append(")")
    lines.append("from sklearn.linear_model import (")
    lines.append("    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,")
    lines.append("    SGDClassifier, SGDRegressor, BayesianRidge, HuberRegressor")
    lines.append(")")
    lines.append("from sklearn.svm import SVC, SVR")
    lines.append("from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor")
    lines.append("from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor")
    lines.append("from sklearn.naive_bayes import GaussianNB")
    lines.append("from sklearn.discriminant_analysis import LinearDiscriminantAnalysis")

    # optional imports
    uses_xgb  = any("xgboost"  in m["name"].lower() for m in models)
    uses_lgbm = any("lightgbm" in m["name"].lower() or "lgbm" in m["name"].lower() for m in models)
    uses_cat  = any("catboost" in m["name"].lower() for m in models)
    if uses_xgb:  lines.append("from xgboost import XGBClassifier, XGBRegressor")
    if uses_lgbm: lines.append("from lightgbm import LGBMClassifier, LGBMRegressor")
    if uses_cat:  lines.append("from catboost import CatBoostClassifier, CatBoostRegressor")

    lines.append("")
    lines.append("# ── Load & preprocess ───────────────────────────────────────")
    lines.append(f'df = pd.read_csv(r"{data_path}")')
    lines.append("X  = df.drop('target', axis=1)")
    lines.append("y  = df['target']")
    lines.append("")
    lines.append("for col in X.columns:")
    lines.append("    if X[col].dtype == 'object':")
    lines.append("        le = LabelEncoder()")
    lines.append("        X[col] = le.fit_transform(X[col].astype(str))")
    lines.append("X = X.fillna(X.median(numeric_only=True))")
    lines.append("")
    if task == "classification":
        lines.append("if y.dtype == 'object':")
        lines.append("    le = LabelEncoder()")
        lines.append("    y  = le.fit_transform(y)")
        lines.append("")
    lines.append("X_train, X_test, y_train, y_test = train_test_split(")
    lines.append("    X, y, test_size=0.2, random_state=42")
    lines.append(")")
    lines.append("")
    lines.append("all_results = {}")
    lines.append("")

    for m in models:
        name   = m["name"]
        params = m.get("best_params", {})
        score  = m.get("score")
        nlo    = name.lower().strip()
        var    = "model_" + nlo.replace(" ", "_").replace("-", "_")
        pvar   = "preds_" + nlo.replace(" ", "_").replace("-", "_")
        svar   = "score_" + nlo.replace(" ", "_").replace("-", "_")

        lines.append(f"# ── {name} {'─'*(48 - len(name))}")
        lines.append(f"# Optuna best {metric}: {score:.4f}")
        if params:
            lines.append("# Best hyperparameters:")
            for k, v in params.items():
                lines.append(f"#   {k:28s} = {v}")
        else:
            lines.append("# No tunable hyperparameters")

        param_str = ", ".join([f"{k}={repr(v)}" for k, v in params.items()])
        if param_str: param_str += ", "

        # Build constructor
        is_cls = (task == "classification")
        if   "random forest"    in nlo: cls = "RandomForestClassifier"    if is_cls else "RandomForestRegressor";    lines.append(f"{var} = {cls}({param_str}random_state=42, n_jobs=-1)")
        elif "xgboost"          in nlo: cls = "XGBClassifier"             if is_cls else "XGBRegressor";             lines.append(f"{var} = {cls}({param_str}random_state=42, verbosity=0)")
        elif "lightgbm" in nlo or "lgbm" in nlo: cls = "LGBMClassifier"  if is_cls else "LGBMRegressor";            lines.append(f"{var} = {cls}({param_str}random_state=42, verbose=-1)")
        elif "catboost"         in nlo: cls = "CatBoostClassifier"        if is_cls else "CatBoostRegressor";        lines.append(f"{var} = {cls}({param_str}random_state=42, verbose=0)")
        elif "gradient boosting"in nlo: cls = "GradientBoostingClassifier"if is_cls else "GradientBoostingRegressor";lines.append(f"{var} = {cls}({param_str}random_state=42)")
        elif "adaboost"         in nlo: cls = "AdaBoostClassifier"        if is_cls else "AdaBoostRegressor";        lines.append(f"{var} = {cls}({param_str}random_state=42)")
        elif "extra trees"      in nlo: cls = "ExtraTreesClassifier"      if is_cls else "ExtraTreesRegressor";      lines.append(f"{var} = {cls}({param_str}random_state=42, n_jobs=-1)")
        elif "bagging"          in nlo: cls = "BaggingClassifier"         if is_cls else "BaggingRegressor";         lines.append(f"{var} = {cls}({param_str}random_state=42)")
        elif "decision tree"    in nlo: cls = "DecisionTreeClassifier"    if is_cls else "DecisionTreeRegressor";    lines.append(f"{var} = {cls}({param_str}random_state=42)")
        elif "knn" in nlo or "k-nearest" in nlo or "k nearest" in nlo:                                               lines.append(f"{var} = {'KNeighborsClassifier' if is_cls else 'KNeighborsRegressor'}({param_str})")
        elif "svm"              in nlo: cls = "SVC"                       if is_cls else "SVR";                      lines.append(f"{var} = {cls}({param_str})")
        elif "logistic"         in nlo:                                                                               lines.append(f"{var} = LogisticRegression({param_str}max_iter=1000, random_state=42)")
        elif nlo == "linear regression":                                                                              lines.append(f"{var} = LinearRegression()")
        elif "ridge"            in nlo:                                                                               lines.append(f"{var} = Ridge({param_str})")
        elif "lasso"            in nlo:                                                                               lines.append(f"{var} = Lasso({param_str}max_iter=5000)")
        elif "elastic net" in nlo or "elasticnet" in nlo:                                                            lines.append(f"{var} = ElasticNet({param_str}max_iter=5000)")
        elif "sgd"              in nlo: cls = "SGDClassifier"             if is_cls else "SGDRegressor";             lines.append(f"{var} = {cls}({param_str}random_state=42)")
        elif "bayesian"         in nlo:                                                                               lines.append(f"{var} = BayesianRidge({param_str})")
        elif "huber"            in nlo:                                                                               lines.append(f"{var} = HuberRegressor({param_str})")
        elif "naive bayes"      in nlo:                                                                               lines.append(f"{var} = GaussianNB({param_str})")
        elif "lda"              in nlo:                                                                               lines.append(f"{var} = LinearDiscriminantAnalysis()")
        else:                                                                                                         lines.append(f"# Skipped: {name} (no constructor mapping found)")

        lines.append(f"{var}.fit(X_train, y_train)")
        lines.append(f"{pvar} = {var}.predict(X_test)")
        if task == "classification":
            lines.append(f"{svar} = accuracy_score(y_test, {pvar})")
            lines.append(f'all_results["{name}"] = {{"metric": "Accuracy", "score": {svar}}}')
            lines.append(f'print(f"{name}: Accuracy = {{{svar}:.4f}}")')
        else:
            lines.append(f"{svar} = float(np.sqrt(mean_squared_error(y_test, {pvar})))")
            lines.append(f'all_results["{name}"] = {{"metric": "RMSE", "score": {svar}}}')
            lines.append(f'print(f"{name}: RMSE = {{{svar}:.4f}}")')
        lines.append("")

    lines.append("# ── Final summary ───────────────────────────────────────────")
    lines.append('print("\\n" + "="*55)')
    lines.append('print("FINAL RESULTS SUMMARY")')
    lines.append('print("="*55)')
    lines.append("for model_name, res in all_results.items():")
    lines.append("    print(f\"  {model_name:30s} {res['metric']}: {res['score']:.4f}\")")
    lines.append("")
    if task == "classification":
        lines.append("best = max(all_results, key=lambda k: all_results[k]['score'])")
    else:
        lines.append("best = min(all_results, key=lambda k: all_results[k]['score'])")
    lines.append("print(f\"\\nBest model : {best}\")")
    lines.append("print(f\"Score      : {all_results[best]['score']:.4f}\")")

    return "\n".join(lines)


def run_automl(data_path: str, selected_models: list, result_path: str,
               n_trials: int = 25, progress_callback=None) -> dict:

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return {"error": f"Could not load dataset: {str(e)}"}

    if "target" not in df.columns:
        return {"error": "Dataset must contain a 'target' column."}

    X = df.drop("target", axis=1)
    y = df["target"]
    X = preprocess(X)

    if y.dtype == "object":
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(y), name="target")

    task = detect_task(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {
        "task":               task,
        "models":             [],
        "dataset_shape":      list(df.shape),
        "n_trials_per_model": n_trials,
    }

    total = len(selected_models)

    for i, model_name in enumerate(selected_models):
        name = _resolve_model_name(model_name)

        # Check optional library availability
        if "xgboost"  in name and not XGBOOST_AVAILABLE:
            results["models"].append({"name": model_name, "score": None, "best_params": {},
                                      "error": "xgboost not installed. Run: pip install xgboost"}); continue
        if ("lightgbm" in name or "lgbm" in name) and not LIGHTGBM_AVAILABLE:
            results["models"].append({"name": model_name, "score": None, "best_params": {},
                                      "error": "lightgbm not installed. Run: pip install lightgbm"}); continue
        if "catboost" in name and not CATBOOST_AVAILABLE:
            results["models"].append({"name": model_name, "score": None, "best_params": {},
                                      "error": "catboost not installed. Run: pip install catboost"}); continue

        # Skip models incompatible with detected task
        cls_only = ["logistic regression", "naive bayes", "lda", "linear discriminant"]
        reg_only = ["linear regression", "ridge regression", "lasso regression",
                    "elastic net", "bayesian ridge", "huber"]
        if task == "regression"     and any(x in name for x in cls_only): continue
        if task == "classification" and any(x in name for x in reg_only): continue

        try:
            objective = _make_objective(name, task, X_train, X_test, y_train, y_test)
            study = optuna.create_study(
                direction="maximize" if task == "classification" else "minimize",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            results["models"].append({
                "name":        model_name,
                "score":       round(study.best_value, 6),
                "best_params": study.best_params,
                "n_trials":    n_trials,
            })

        except Exception as e:
            results["models"].append({
                "name": model_name, "score": None,
                "best_params": {}, "error": str(e)
            })

        if progress_callback:
            progress_callback(model_name, i + 1, total)

    # Sort leaderboard
    valid   = [m for m in results["models"] if m.get("score") is not None]
    invalid = [m for m in results["models"] if m.get("score") is None]
    valid.sort(key=lambda x: x["score"], reverse=(task == "classification"))
    results["models"] = valid + invalid

    # Generate final code
    results["final_code"] = _generate_final_code(results, data_path)

    # Persist
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    return results