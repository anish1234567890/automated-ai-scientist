import pandas as pd
import numpy as np
import optuna
import json
import os
from optuna.pruners import MedianPruner
from optuna.trial import FixedTrial
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import (
    VotingClassifier, VotingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    BaggingClassifier, BaggingRegressor,
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor,
    BayesianRidge, HuberRegressor,
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


# ── Model registry ─────────────────────────────────────────────────
MODEL_REGISTRY = {
    "random forest":       (RandomForestClassifier,       RandomForestRegressor,       "both"),
    "xgboost":             (XGBClassifier,                XGBRegressor,                "both"),
    "lightgbm":            (LGBMClassifier if LIGHTGBM_AVAILABLE else None,
                            LGBMRegressor  if LIGHTGBM_AVAILABLE else None,            "both"),
    "catboost":            (CatBoostClassifier if CATBOOST_AVAILABLE else None,
                            CatBoostRegressor  if CATBOOST_AVAILABLE else None,        "both"),
    "gradient boosting":   (GradientBoostingClassifier,   GradientBoostingRegressor,   "both"),
    "adaboost":            (AdaBoostClassifier,            AdaBoostRegressor,           "both"),
    "extra trees":         (ExtraTreesClassifier,          ExtraTreesRegressor,         "both"),
    "bagging":             (BaggingClassifier,             BaggingRegressor,            "both"),
    "decision tree":       (DecisionTreeClassifier,        DecisionTreeRegressor,       "both"),
    "knn":                 (KNeighborsClassifier,          KNeighborsRegressor,         "both"),
    "k-nearest neighbors": (KNeighborsClassifier,          KNeighborsRegressor,         "both"),
    "svm":                 (SVC,                           SVR,                         "both"),
    "logistic regression": (LogisticRegression,            None,                        "classification"),
    "linear regression":   (None,                          LinearRegression,            "regression"),
    "ridge regression":    (None,                          Ridge,                       "regression"),
    "lasso regression":    (None,                          Lasso,                       "regression"),
    "elastic net":         (None,                          ElasticNet,                  "regression"),
    "sgd":                 (SGDClassifier,                 SGDRegressor,                "both"),
    "bayesian ridge":      (None,                          BayesianRidge,               "regression"),
    "huber":               (None,                          HuberRegressor,              "regression"),
    "naive bayes":         (GaussianNB,                    None,                        "classification"),
    "lda":                 (LinearDiscriminantAnalysis,    None,                        "classification"),
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
    n = user_name.lower().strip()
    if n in MODEL_REGISTRY:
        return n
    for key in MODEL_REGISTRY:
        if key in n or n in key:
            return key
    return n


def _make_objective(model_name: str, task: str, X: pd.DataFrame, y: pd.Series):
    """
    Optuna objective using 5-fold CV inside a Pipeline.
    Pipeline = PolynomialFeatures(interaction_only) → SelectKBest → model.
    interaction_only=True avoids feature explosion on wide datasets.
    k is 30% of original feature count (at least 1).
    This structure prevents data leakage — feature selection happens
    inside each CV fold, not before splitting.
    """
    name = model_name.lower().strip()
    k    = max(1, min(int(0.3 * X.shape[1]), X.shape[1]))

    def objective(trial):
        model = _build_model(trial, name, task)
        if model is None:
            return 0.0 if task == "classification" else 1e9

        pipeline = Pipeline([
            ("poly",   PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
            ("select", SelectKBest(f_classif if task == "classification" else f_regression, k=k)),
            ("model",  model),
        ])

        try:
            if task == "classification":
                return float(cross_val_score(
                    pipeline, X, y, cv=5, scoring="accuracy", n_jobs=-1).mean())
            else:
                # neg_root_mean_squared_error → negate to get positive RMSE
                return float(-cross_val_score(
                    pipeline, X, y, cv=5,
                    scoring="neg_root_mean_squared_error", n_jobs=-1).mean())
        except Exception:
            return 0.0 if task == "classification" else 1e9

    return objective


def _build_model(trial, name: str, task: str):
    """Build sklearn model with Optuna-suggested hyperparameters."""
    is_cls = (task == "classification")

    if "random forest" in name:
        return (RandomForestClassifier if is_cls else RandomForestRegressor)(
            n_estimators     =trial.suggest_int("n_estimators", 50, 400),
            max_depth        =trial.suggest_int("max_depth", 3, 25),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf =trial.suggest_int("min_samples_leaf", 1, 8),
            max_features     =trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=42, n_jobs=-1,
        )

    elif "xgboost" in name:
        if not XGBOOST_AVAILABLE: return None
        return (XGBClassifier if is_cls else XGBRegressor)(
            n_estimators    =trial.suggest_int("n_estimators", 50, 400),
            learning_rate   =trial.suggest_float("learning_rate", 0.01, 0.3),
            max_depth       =trial.suggest_int("max_depth", 3, 12),
            subsample       =trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha       =trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            reg_lambda      =trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
            random_state=42, verbosity=0, n_jobs=-1,
        )

    elif "lightgbm" in name or "lgbm" in name:
        if not LIGHTGBM_AVAILABLE: return None
        return (LGBMClassifier if is_cls else LGBMRegressor)(
            n_estimators    =trial.suggest_int("n_estimators", 50, 400),
            learning_rate   =trial.suggest_float("learning_rate", 0.01, 0.3),
            max_depth       =trial.suggest_int("max_depth", 3, 12),
            num_leaves      =trial.suggest_int("num_leaves", 20, 150),
            subsample       =trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            random_state=42, n_jobs=-1, verbose=-1,
        )

    elif "catboost" in name:
        if not CATBOOST_AVAILABLE: return None
        return (CatBoostClassifier if is_cls else CatBoostRegressor)(
            iterations   =trial.suggest_int("iterations", 50, 400),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            depth        =trial.suggest_int("depth", 3, 10),
            l2_leaf_reg  =trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            random_state=42, verbose=0,
        )

    elif "gradient boosting" in name:
        return (GradientBoostingClassifier if is_cls else GradientBoostingRegressor)(
            n_estimators =trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            max_depth    =trial.suggest_int("max_depth", 2, 8),
            subsample    =trial.suggest_float("subsample", 0.5, 1.0),
            max_features =trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=42,
        )

    elif "adaboost" in name:
        return (AdaBoostClassifier if is_cls else AdaBoostRegressor)(
            n_estimators =trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 2.0),
            random_state=42,
        )

    elif "extra trees" in name:
        return (ExtraTreesClassifier if is_cls else ExtraTreesRegressor)(
            n_estimators     =trial.suggest_int("n_estimators", 50, 400),
            max_depth        =trial.suggest_int("max_depth", 3, 25),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            max_features     =trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=42, n_jobs=-1,
        )

    elif "bagging" in name:
        return (BaggingClassifier if is_cls else BaggingRegressor)(
            n_estimators=trial.suggest_int("n_estimators", 10, 100),
            max_samples =trial.suggest_float("max_samples", 0.5, 1.0),
            max_features=trial.suggest_float("max_features", 0.5, 1.0),
            random_state=42, n_jobs=-1,
        )

    elif "decision tree" in name:
        return (DecisionTreeClassifier if is_cls else DecisionTreeRegressor)(
            max_depth        =trial.suggest_int("max_depth", 2, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf =trial.suggest_int("min_samples_leaf", 1, 10),
            criterion        =trial.suggest_categorical(
                "criterion", ["gini", "entropy"] if is_cls else ["squared_error", "friedman_mse"]),
            random_state=42,
        )

    elif "knn" in name or "k-nearest" in name or "k nearest" in name:
        return (KNeighborsClassifier if is_cls else KNeighborsRegressor)(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 30),
            weights    =trial.suggest_categorical("weights", ["uniform", "distance"]),
            metric     =trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
        )

    elif "svm" in name:
        C      = trial.suggest_float("C", 0.01, 100.0, log=True)
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
        if is_cls:
            return SVC(C=C, kernel=kernel,
                       gamma=trial.suggest_categorical("gamma", ["scale", "auto"]))
        else:
            return SVR(C=C, kernel=kernel,
                       epsilon=trial.suggest_float("epsilon", 0.01, 1.0))

    elif "logistic" in name:
        if not is_cls: return None
        penalty  = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        solver   = "saga" if penalty in ["l1", "elasticnet"] else "lbfgs"
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else None
        return LogisticRegression(
            C=trial.suggest_float("C", 0.01, 100.0, log=True),
            penalty=penalty, solver=solver, l1_ratio=l1_ratio,
            max_iter=1000, random_state=42,
        )

    elif name == "linear regression":
        return LinearRegression()

    elif "ridge" in name:
        return Ridge(alpha=trial.suggest_float("alpha", 0.01, 100.0, log=True))

    elif "lasso" in name:
        return Lasso(alpha=trial.suggest_float("alpha", 0.0001, 10.0, log=True), max_iter=5000)

    elif "elastic net" in name or "elasticnet" in name:
        return ElasticNet(
            alpha   =trial.suggest_float("alpha", 0.0001, 10.0, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
            max_iter=5000,
        )

    elif "sgd" in name:
        alpha = trial.suggest_float("alpha", 1e-5, 1.0, log=True)
        lr    = trial.suggest_categorical("learning_rate", ["optimal", "invscaling", "adaptive"])
        if is_cls:
            return SGDClassifier(
                alpha=alpha, learning_rate=lr,
                loss=trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber"]),
                random_state=42, max_iter=1000,
            )
        else:
            return SGDRegressor(
                alpha=alpha, learning_rate=lr,
                loss=trial.suggest_categorical("loss",
                                               ["squared_error", "huber", "epsilon_insensitive"]),
                random_state=42, max_iter=1000,
            )

    elif "bayesian ridge" in name or "bayesian" in name:
        if is_cls: return None
        return BayesianRidge(
            alpha_1 =trial.suggest_float("alpha_1",  1e-7, 1e-4, log=True),
            alpha_2 =trial.suggest_float("alpha_2",  1e-7, 1e-4, log=True),
            lambda_1=trial.suggest_float("lambda_1", 1e-7, 1e-4, log=True),
            lambda_2=trial.suggest_float("lambda_2", 1e-7, 1e-4, log=True),
        )

    elif "huber" in name:
        if is_cls: return None
        return HuberRegressor(
            epsilon=trial.suggest_float("epsilon", 1.01, 2.0),
            alpha  =trial.suggest_float("alpha", 1e-5, 1.0, log=True),
            max_iter=500,
        )

    elif "naive bayes" in name or "gaussiannb" in name:
        if not is_cls: return None
        return GaussianNB(var_smoothing=trial.suggest_float("var_smoothing", 1e-11, 1e-7, log=True))

    elif "lda" in name or "linear discriminant" in name:
        if not is_cls: return None
        return LinearDiscriminantAnalysis()

    return None


def _build_ensemble(top_models: list, task: str, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Build a Voting ensemble from the top-3 tuned models and evaluate with 5-fold CV.
    Uses soft voting for classification when all estimators support predict_proba.
    """
    estimators = []
    for i, m in enumerate(top_models):
        mdl = _build_model(FixedTrial(m["best_params"]), m["name"].lower().strip(), task)
        if mdl is not None:
            estimators.append((f"model{i}", mdl))

    if len(estimators) < 2:
        return {}

    k = max(1, min(int(0.3 * X.shape[1]), X.shape[1]))

    # Try soft voting first (needs predict_proba); fall back to hard
    for voting in (["soft", "hard"] if task == "classification" else [None]):
        try:
            if task == "classification":
                ensemble = VotingClassifier(estimators=estimators, voting=voting)
            else:
                ensemble = VotingRegressor(estimators=estimators)

            pipeline = Pipeline([
                ("poly",     PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
                ("select",   SelectKBest(f_classif if task == "classification" else f_regression, k=k)),
                ("ensemble", ensemble),
            ])

            if task == "classification":
                cv_score = float(cross_val_score(
                    pipeline, X, y, cv=5, scoring="accuracy", n_jobs=-1).mean())
            else:
                cv_score = float(-cross_val_score(
                    pipeline, X, y, cv=5,
                    scoring="neg_root_mean_squared_error", n_jobs=-1).mean())

            return {
                "models_used": [m["name"] for m in top_models],
                "cv_score":    round(cv_score, 6),
                "metric":      "Accuracy" if task == "classification" else "RMSE",
                "voting":      voting or "averaging",
            }
        except Exception:
            if task != "classification":
                break
            continue  # try hard voting

    return {}


def _generate_final_code(results: dict, data_path: str) -> str:
    task    = results.get("task", "classification")
    models  = [m for m in results.get("models", []) if m.get("score") is not None]
    metric  = "Accuracy" if task == "classification" else "RMSE"
    is_cls  = task == "classification"
    scoring = "accuracy" if is_cls else "neg_root_mean_squared_error"

    lines = [
        "# ================================================================",
        "# AUTO-GENERATED BY AUTOMATED AI SCIENTIST",
        f"# Task  : {task.upper()}",
        f"# Metric: {metric} (5-fold cross-validation)",
        "# Pipeline: PolynomialFeatures → SelectKBest → Model",
        "# ================================================================",
        "",
        "import pandas as pd",
        "import numpy as np",
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "from sklearn.model_selection import cross_val_score",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import LabelEncoder, PolynomialFeatures",
        "from sklearn.feature_selection import SelectKBest, f_classif, f_regression",
        "from sklearn.metrics import accuracy_score, mean_squared_error",
        "from sklearn.ensemble import (",
        "    VotingClassifier, VotingRegressor,",
        "    RandomForestClassifier, RandomForestRegressor,",
        "    GradientBoostingClassifier, GradientBoostingRegressor,",
        "    AdaBoostClassifier, AdaBoostRegressor,",
        "    ExtraTreesClassifier, ExtraTreesRegressor,",
        "    BaggingClassifier, BaggingRegressor,",
        ")",
        "from sklearn.linear_model import (",
        "    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,",
        "    SGDClassifier, SGDRegressor, BayesianRidge, HuberRegressor,",
        ")",
        "from sklearn.svm import SVC, SVR",
        "from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor",
        "from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor",
        "from sklearn.naive_bayes import GaussianNB",
        "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis",
    ]

    if any("xgboost"  in m["name"].lower() for m in models):
        lines.append("from xgboost import XGBClassifier, XGBRegressor")
    if any("lightgbm" in m["name"].lower() or "lgbm" in m["name"].lower() for m in models):
        lines.append("from lightgbm import LGBMClassifier, LGBMRegressor")
    if any("catboost" in m["name"].lower() for m in models):
        lines.append("from catboost import CatBoostClassifier, CatBoostRegressor")

    lines += [
        "",
        "# ── Load & preprocess ───────────────────────────────────────",
        f'df = pd.read_csv(r"{data_path}")',
        "X  = df.drop('target', axis=1)",
        "y  = df['target']",
        "",
        "for col in X.columns:",
        "    if X[col].dtype == 'object':",
        "        X[col] = LabelEncoder().fit_transform(X[col].astype(str))",
        "X = X.fillna(X.median(numeric_only=True))",
    ]
    if is_cls:
        lines += ["if y.dtype == 'object':", "    y = LabelEncoder().fit_transform(y)"]

    lines += [
        "",
        "# Feature pipeline settings",
        "k       = max(1, int(0.3 * X.shape[1]))  # top 30% features",
        f"scoring = '{scoring}'",
        "",
        "all_results = {}",
        "estimators  = []  # for ensemble",
        "",
    ]

    for m in models:
        name  = m["name"]
        params = m.get("best_params", {})
        score  = m.get("score")
        nlo    = name.lower().strip()
        var    = nlo.replace(" ", "_").replace("-", "_")
        ps     = ", ".join([f"{k}={repr(v)}" for k, v in params.items()])
        if ps: ps += ", "

        lines.append(f"# ── {name} {'─'*(46-len(name))}")
        lines.append(f"# CV {metric}: {score:.4f}")

        if   "random forest"     in nlo: cls="RandomForestClassifier"     if is_cls else "RandomForestRegressor";    lines.append(f"mdl_{var} = {cls}({ps}random_state=42, n_jobs=-1)")
        elif "xgboost"           in nlo: cls="XGBClassifier"              if is_cls else "XGBRegressor";             lines.append(f"mdl_{var} = {cls}({ps}random_state=42, verbosity=0)")
        elif "lightgbm" in nlo or "lgbm" in nlo: cls="LGBMClassifier"    if is_cls else "LGBMRegressor";            lines.append(f"mdl_{var} = {cls}({ps}random_state=42, verbose=-1)")
        elif "catboost"          in nlo: cls="CatBoostClassifier"         if is_cls else "CatBoostRegressor";        lines.append(f"mdl_{var} = {cls}({ps}random_state=42, verbose=0)")
        elif "gradient boosting" in nlo: cls="GradientBoostingClassifier" if is_cls else "GradientBoostingRegressor";lines.append(f"mdl_{var} = {cls}({ps}random_state=42)")
        elif "adaboost"          in nlo: cls="AdaBoostClassifier"         if is_cls else "AdaBoostRegressor";        lines.append(f"mdl_{var} = {cls}({ps}random_state=42)")
        elif "extra trees"       in nlo: cls="ExtraTreesClassifier"       if is_cls else "ExtraTreesRegressor";      lines.append(f"mdl_{var} = {cls}({ps}random_state=42, n_jobs=-1)")
        elif "bagging"           in nlo: cls="BaggingClassifier"          if is_cls else "BaggingRegressor";         lines.append(f"mdl_{var} = {cls}({ps}random_state=42)")
        elif "decision tree"     in nlo: cls="DecisionTreeClassifier"     if is_cls else "DecisionTreeRegressor";    lines.append(f"mdl_{var} = {cls}({ps}random_state=42)")
        elif "knn" in nlo or "k-nearest" in nlo:                                                                     lines.append(f"mdl_{var} = {'KNeighborsClassifier' if is_cls else 'KNeighborsRegressor'}({ps})")
        elif "svm"               in nlo: cls="SVC"                        if is_cls else "SVR";                      lines.append(f"mdl_{var} = {cls}({ps})")
        elif "logistic"          in nlo:                                                                              lines.append(f"mdl_{var} = LogisticRegression({ps}max_iter=1000, random_state=42)")
        elif nlo == "linear regression":                                                                              lines.append(f"mdl_{var} = LinearRegression()")
        elif "ridge"             in nlo:                                                                              lines.append(f"mdl_{var} = Ridge({ps})")
        elif "lasso"             in nlo:                                                                              lines.append(f"mdl_{var} = Lasso({ps}max_iter=5000)")
        elif "elastic net" in nlo or "elasticnet" in nlo:                                                            lines.append(f"mdl_{var} = ElasticNet({ps}max_iter=5000)")
        elif "sgd"               in nlo: cls="SGDClassifier"              if is_cls else "SGDRegressor";             lines.append(f"mdl_{var} = {cls}({ps}random_state=42)")
        elif "bayesian"          in nlo:                                                                              lines.append(f"mdl_{var} = BayesianRidge({ps})")
        elif "huber"             in nlo:                                                                              lines.append(f"mdl_{var} = HuberRegressor({ps})")
        elif "naive bayes"       in nlo:                                                                              lines.append(f"mdl_{var} = GaussianNB({ps})")
        elif "lda"               in nlo:                                                                              lines.append(f"mdl_{var} = LinearDiscriminantAnalysis()")
        else:                                                                                                         lines.append(f"# Skipped: {name}")

        lines += [
            f"pipe_{var} = Pipeline([",
            f"    ('poly',   PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),",
            f"    ('select', SelectKBest(f_classif if {is_cls} else f_regression, k=k)),",
            f"    ('model',  mdl_{var}),",
            f"])",
            f"cv_{var} = cross_val_score(pipe_{var}, X, y, cv=5, scoring=scoring, n_jobs=-1).mean()",
        ]
        if is_cls:
            lines += [
                f'all_results["{name}"] = {{"metric": "Accuracy (CV-5)", "score": cv_{var}}}',
                f'print(f"{name}: CV Accuracy = {{cv_{var}:.4f}}")',
                f'estimators.append(("{var}", mdl_{var}))',
            ]
        else:
            lines += [
                f'all_results["{name}"] = {{"metric": "RMSE (CV-5)", "score": -cv_{var}}}',
                f'print(f"{name}: CV RMSE = {{-cv_{var}:.4f}}")',
                f'estimators.append(("{var}", mdl_{var}))',
            ]
        lines.append("")

    # Ensemble
    ensemble_info = results.get("ensemble", {})
    if ensemble_info and not ensemble_info.get("error") and len(models) >= 2:
        ev = ensemble_info.get("cv_score", "N/A")
        lines += [
            "# ── Ensemble (Top-3 Voting) ─────────────────────────────",
            f"# Ensemble CV {metric}: {ev}",
            "top3 = estimators[:3]",
        ]
        if is_cls:
            lines += [
                "try:",
                "    voting_clf = VotingClassifier(estimators=top3, voting='soft')",
                "    pipe_ens   = Pipeline([",
                "        ('poly',   PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),",
                "        ('select', SelectKBest(f_classif, k=k)),",
                "        ('ens',    voting_clf),",
                "    ])",
                "    ens_score = cross_val_score(pipe_ens, X, y, cv=5, scoring=scoring, n_jobs=-1).mean()",
                "    all_results['Ensemble (Top-3)'] = {'metric': 'Accuracy (CV-5)', 'score': ens_score}",
                "    print(f'Ensemble (Top-3): CV Accuracy = {ens_score:.4f}')",
                "except Exception as e:",
                "    print(f'Ensemble failed: {e}')",
            ]
        else:
            lines += [
                "try:",
                "    voting_reg = VotingRegressor(estimators=top3)",
                "    pipe_ens   = Pipeline([",
                "        ('poly',   PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),",
                "        ('select', SelectKBest(f_regression, k=k)),",
                "        ('ens',    voting_reg),",
                "    ])",
                "    ens_score = -cross_val_score(pipe_ens, X, y, cv=5, scoring=scoring, n_jobs=-1).mean()",
                "    all_results['Ensemble (Top-3)'] = {'metric': 'RMSE (CV-5)', 'score': ens_score}",
                "    print(f'Ensemble (Top-3): CV RMSE = {ens_score:.4f}')",
                "except Exception as e:",
                "    print(f'Ensemble failed: {e}')",
            ]
        lines.append("")

    lines += [
        "# ── Summary ─────────────────────────────────────────────────",
        'print("\\n" + "="*55)',
        'print("FINAL RESULTS (5-fold CV)")',
        'print("="*55)',
        "for mn, res in all_results.items():",
        "    print(f\"  {mn:35s} {res['metric']}: {res['score']:.4f}\")",
        "",
    ]
    if is_cls:
        lines.append("best = max(all_results, key=lambda k: all_results[k]['score'])")
    else:
        lines.append("best = min(all_results, key=lambda k: all_results[k]['score'])")
    lines += [
        "print(f\"\\nBest model : {best}\")",
        "print(f\"CV Score   : {all_results[best]['score']:.4f}\")",
    ]

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

    results = {
        "task":               task,
        "models":             [],
        "dataset_shape":      list(df.shape),
        "n_trials_per_model": n_trials,
    }

    total = len(selected_models)

    for i, model_name in enumerate(selected_models):
        name = _resolve_model_name(model_name)

        if "xgboost"  in name and not XGBOOST_AVAILABLE:
            results["models"].append({"name": model_name, "score": None, "best_params": {},
                                      "error": "xgboost not installed. Run: pip install xgboost"}); continue
        if ("lightgbm" in name or "lgbm" in name) and not LIGHTGBM_AVAILABLE:
            results["models"].append({"name": model_name, "score": None, "best_params": {},
                                      "error": "lightgbm not installed. Run: pip install lightgbm"}); continue
        if "catboost" in name and not CATBOOST_AVAILABLE:
            results["models"].append({"name": model_name, "score": None, "best_params": {},
                                      "error": "catboost not installed. Run: pip install catboost"}); continue

        cls_only = ["logistic regression", "naive bayes", "lda", "linear discriminant"]
        reg_only = ["linear regression", "ridge regression", "lasso regression",
                    "elastic net", "bayesian ridge", "huber"]
        if task == "regression"     and any(x in name for x in cls_only): continue
        if task == "classification" and any(x in name for x in reg_only): continue

        try:
            objective = _make_objective(name, task, X, y)
            study = optuna.create_study(
                direction="maximize",
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
                sampler=optuna.samplers.TPESampler(seed=42),
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
                "best_params": {}, "error": str(e),
            })

        if progress_callback:
            progress_callback(model_name, i + 1, total)

    # Sort leaderboard
    valid   = [m for m in results["models"] if m.get("score") is not None]
    invalid = [m for m in results["models"] if m.get("score") is None]
    valid.sort(key=lambda x: x["score"], reverse=(task == "classification"))
    results["models"] = valid + invalid

    # Ensemble
    if len(valid) >= 2:
        ensemble_result = _build_ensemble(valid[:3], task, X, y)
        if ensemble_result:
            results["ensemble"] = ensemble_result

    results["final_code"] = _generate_final_code(results, data_path)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    return results