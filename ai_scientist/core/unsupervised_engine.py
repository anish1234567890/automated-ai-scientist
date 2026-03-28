"""
unsupervised_engine.py
──────────────────────
Clustering + anomaly detection engine for the Automated AI Scientist.
No target column required.

Algorithms:
  Clustering        : KMeans, DBSCAN, Agglomerative, GaussianMixture
  Anomaly Detection : IsolationForest, LocalOutlierFactor

Metrics (no labels needed):
  Silhouette Score     — higher is better  (+1 perfect, 0 overlap, -1 wrong)
  Davies-Bouldin Index — lower is better   (0 = perfect)
  Calinski-Harabasz    — higher is better
"""

import pandas as pd
import numpy as np
import optuna
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster       import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture       import GaussianMixture
from sklearn.ensemble      import IsolationForest
from sklearn.neighbors     import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics       import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Detect if unsupervised mode should activate ───────────────────
def should_run_unsupervised(df: pd.DataFrame, user_prompt: str) -> bool:
    keywords = [
        "cluster", "clustering", "group", "segment", "unsupervised",
        "pattern", "anomaly", "outlier", "find groups", "pca",
        "dimensionality", "explore", "discover", "no label"
    ]
    has_keyword = any(kw in user_prompt.lower() for kw in keywords)
    has_target  = "target" in df.columns
    return has_keyword or not has_target


# ── Preprocessing ─────────────────────────────────────────────────
def preprocess_unsupervised(df: pd.DataFrame):
    X = df.drop("target", axis=1, errors="ignore").copy()
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    X = X.fillna(X.median(numeric_only=True))
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, list(X.columns)


# ── PCA helpers ───────────────────────────────────────────────────
def compute_pca_2d(X: np.ndarray) -> list:
    n = min(2, X.shape[1])
    pca    = PCA(n_components=n, random_state=42)
    coords = pca.fit_transform(X)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])
    return coords.tolist()


def compute_pca_variance(X: np.ndarray) -> list:
    n   = min(X.shape[1], 10)
    pca = PCA(n_components=n, random_state=42)
    pca.fit(X)
    return [round(float(v), 4) for v in pca.explained_variance_ratio_]


# ── Score a clustering result ─────────────────────────────────────
def _score(X: np.ndarray, labels: np.ndarray) -> dict:
    unique = np.unique(labels)
    mask   = labels != -1
    n_valid = len(np.unique(labels[mask]))
    out = {
        "silhouette":        None,
        "davies_bouldin":    None,
        "calinski_harabasz": None,
        "n_clusters_found":  int(len(unique[unique != -1])),
        "n_noise_points":    int(np.sum(labels == -1)),
    }
    if n_valid < 2 or np.sum(mask) < 2:
        return out
    try:
        out["silhouette"]        = round(float(silhouette_score(X[mask], labels[mask])), 4)
        out["davies_bouldin"]    = round(float(davies_bouldin_score(X[mask], labels[mask])), 4)
        out["calinski_harabasz"] = round(float(calinski_harabasz_score(X[mask], labels[mask])), 4)
    except Exception:
        pass
    return out


# ── Algorithm runners ─────────────────────────────────────────────
def _run_kmeans(X, n_trials):
    best = {"score": -999, "params": {}, "labels": np.zeros(len(X), dtype=int)}

    def obj(trial):
        k     = trial.suggest_int("n_clusters", 2, min(15, len(X) // 2))
        init  = trial.suggest_categorical("init", ["k-means++", "random"])
        n_ini = trial.suggest_int("n_init", 5, 20)
        lbl   = KMeans(n_clusters=k, init=init, n_init=n_ini,
                       random_state=42, max_iter=300).fit_predict(X)
        sc    = _score(X, lbl)
        sil = sc["silhouette"] or -1.0
        db  = sc["davies_bouldin"] or 10
        ch  = sc["calinski_harabasz"] or 0

        score = (
            0.5 * sil
            - 0.3 * db
            + 0.2 * ch
        )

        if score > best["score"]:
            best.update({"score": score,
                         "params": {"n_clusters": k, "init": init, "n_init": n_ini},
                         "labels": lbl})
        return score

    optuna.create_study(direction="maximize",
                        sampler=optuna.samplers.TPESampler(seed=42)
                        ).optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return {"name": "K-Means", "best_params": best["params"],
            "labels": best["labels"].tolist(), **_score(X, best["labels"])}


def _run_dbscan(X, n_trials):
    best = {"score": -999, "params": {}, "labels": np.full(len(X), -1, dtype=int)}

    def obj(trial):
        eps  = trial.suggest_float("eps", 0.1, 5.0)
        msp  = trial.suggest_int("min_samples", 2, 20)
        lbl  = DBSCAN(eps=eps, min_samples=msp).fit_predict(X)
        sc   = _score(X, lbl)
        sil = sc["silhouette"] or -1.0
        db  = sc["davies_bouldin"] or 10
        ch  = sc["calinski_harabasz"] or 0

        score = (
            0.5 * sil
            - 0.3 * db
            + 0.2 * ch
        )

        if score > best["score"]:
            best.update({"score": score,
                         "params": {"eps": round(eps, 3), "min_samples": msp},
                         "labels": lbl})
        return score

    optuna.create_study(direction="maximize",
                        sampler=optuna.samplers.TPESampler(seed=42)
                        ).optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return {"name": "DBSCAN", "best_params": best["params"],
            "labels": best["labels"].tolist(), **_score(X, best["labels"])}


def _run_agglomerative(X, n_trials):
    best = {"score": -999, "params": {}, "labels": np.zeros(len(X), dtype=int)}

    def obj(trial):
        k    = trial.suggest_int("n_clusters", 2, min(15, len(X) // 2))
        link = trial.suggest_categorical("linkage",
                                         ["ward", "complete", "average", "single"])
        lbl  = AgglomerativeClustering(n_clusters=k, linkage=link).fit_predict(X)
        sc   = _score(X, lbl)
        sil = sc["silhouette"] or -1.0
        db  = sc["davies_bouldin"] or 10
        ch  = sc["calinski_harabasz"] or 0

        score = (
            0.5 * sil
            - 0.3 * db
            + 0.2 * ch
        )

        if score > best["score"]:
            best.update({"score": score,
                         "params": {"n_clusters": k, "linkage": link},
                         "labels": lbl})
        return score

    optuna.create_study(direction="maximize",
                        sampler=optuna.samplers.TPESampler(seed=42)
                        ).optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return {"name": "Agglomerative", "best_params": best["params"],
            "labels": best["labels"].tolist(), **_score(X, best["labels"])}


def _run_gmm(X, n_trials):
    best = {"score": -999, "params": {}, "labels": np.zeros(len(X), dtype=int)}

    def obj(trial):
        k     = trial.suggest_int("n_components", 2, min(15, len(X) // 2))
        ctype = trial.suggest_categorical("covariance_type",
                                          ["full", "tied", "diag", "spherical"])
        model = GaussianMixture(n_components=k, covariance_type=ctype,
                                random_state=42, max_iter=200)
        lbl   = model.fit_predict(X)
        sc    = _score(X, lbl)
        sil = sc["silhouette"] or -1.0
        db  = sc["davies_bouldin"] or 10
        ch  = sc["calinski_harabasz"] or 0

        score = (
            0.5 * sil
            - 0.3 * db
            + 0.2 * ch
        )

        if score > best["score"]:
            best.update({"score": score,
                         "params": {"n_components": k, "covariance_type": ctype},
                         "labels": lbl})
        return score

    optuna.create_study(direction="maximize",
                        sampler=optuna.samplers.TPESampler(seed=42)
                        ).optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return {"name": "Gaussian Mixture", "best_params": best["params"],
            "labels": best["labels"].tolist(), **_score(X, best["labels"])}


def _run_isolation_forest(X, n_trials):
    best = {"score": -999, "params": {}, "labels": np.zeros(len(X), dtype=int)}

    def obj(trial):
        n_est  = trial.suggest_int("n_estimators", 50, 300)
        contam = trial.suggest_float("contamination", 0.01, 0.3)
        raw    = IsolationForest(n_estimators=n_est, contamination=contam,
                                  random_state=42, n_jobs=-1).fit_predict(X)
        lbl    = np.where(raw == 1, 0, 1)
        sc     = _score(X, lbl)
        sil = sc["silhouette"] or -1.0
        db  = sc["davies_bouldin"] or 10
        ch  = sc["calinski_harabasz"] or 0

        score = (
            0.5 * sil
            - 0.3 * db
            + 0.2 * ch
        )

        if score > best["score"]:
            best.update({"score": score,
                         "params": {"n_estimators": n_est,
                                    "contamination": round(contam, 3),
                                    "anomalies_found": int(np.sum(lbl == 1))},
                         "labels": lbl})
        return score

    optuna.create_study(direction="maximize",
                        sampler=optuna.samplers.TPESampler(seed=42)
                        ).optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return {"name": "Isolation Forest", "best_params": best["params"],
            "labels": best["labels"].tolist(), **_score(X, best["labels"])}


def _run_lof(X, n_trials):
    best = {"score": -999, "params": {}, "labels": np.zeros(len(X), dtype=int)}

    def obj(trial):
        n_nbr  = trial.suggest_int("n_neighbors", 5, 50)
        contam = trial.suggest_float("contamination", 0.01, 0.3)
        raw    = LocalOutlierFactor(n_neighbors=n_nbr,
                                    contamination=contam, n_jobs=-1).fit_predict(X)
        lbl    = np.where(raw == 1, 0, 1)
        sc     = _score(X, lbl)
        sil = sc["silhouette"] or -1.0
        db  = sc["davies_bouldin"] or 10
        ch  = sc["calinski_harabasz"] or 0

        score = (
            0.5 * sil
            - 0.3 * db
            + 0.2 * ch
        )

        if score > best["score"]:
            best.update({"score": score,
                         "params": {"n_neighbors": n_nbr,
                                    "contamination": round(contam, 3),
                                    "outliers_found": int(np.sum(lbl == 1))},
                         "labels": lbl})
        return score

    optuna.create_study(direction="maximize",
                        sampler=optuna.samplers.TPESampler(seed=42)
                        ).optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return {"name": "Local Outlier Factor", "best_params": best["params"],
            "labels": best["labels"].tolist(), **_score(X, best["labels"])}


# ── Dispatcher ────────────────────────────────────────────────────
ALGO_MAP = {
    "k-means":              _run_kmeans,
    "kmeans":               _run_kmeans,
    "dbscan":               _run_dbscan,
    "agglomerative":        _run_agglomerative,
    "hierarchical":         _run_agglomerative,
    "gaussian mixture":     _run_gmm,
    "gmm":                  _run_gmm,
    "isolation forest":     _run_isolation_forest,
    "anomaly detection":    _run_isolation_forest,
    "local outlier factor": _run_lof,
    "lof":                  _run_lof,
}

def _resolve_algo(name: str):
    n = name.lower().strip()
    if n in ALGO_MAP: return ALGO_MAP[n]
    for key in ALGO_MAP:
        if key in n or n in key:
            return ALGO_MAP[key]
    return None


# ── Code generator ────────────────────────────────────────────────
def _generate_unsupervised_code(results: dict, data_path: str) -> str:
    clustering = [c for c in results.get("clustering", [])
                  if c.get("silhouette") is not None]
    lines = []
    lines.append("# ================================================================")
    lines.append("# AUTO-GENERATED BY AUTOMATED AI SCIENTIST (UNSUPERVISED)")
    lines.append("# Hyperparameters tuned by Optuna TPE sampler")
    lines.append("# ================================================================")
    lines.append("")
    lines.append("import pandas as pd, numpy as np, matplotlib.pyplot as plt, warnings")
    lines.append("warnings.filterwarnings('ignore')")
    lines.append("from sklearn.preprocessing import LabelEncoder, StandardScaler")
    lines.append("from sklearn.decomposition import PCA")
    lines.append("from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering")
    lines.append("from sklearn.mixture import GaussianMixture")
    lines.append("from sklearn.ensemble import IsolationForest")
    lines.append("from sklearn.neighbors import LocalOutlierFactor")
    lines.append("from sklearn.metrics import silhouette_score, davies_bouldin_score")
    lines.append("")
    lines.append(f'df       = pd.read_csv(r"{data_path}")')
    lines.append("X        = df.drop('target', axis=1, errors='ignore')")
    lines.append("for col in X.columns:")
    lines.append("    if X[col].dtype == 'object':")
    lines.append("        X[col] = LabelEncoder().fit_transform(X[col].astype(str))")
    lines.append("X        = X.fillna(X.median(numeric_only=True))")
    lines.append("X_scaled = StandardScaler().fit_transform(X)")
    lines.append("X_2d     = PCA(n_components=2, random_state=42).fit_transform(X_scaled)")
    lines.append("")
    lines.append("all_results = {}")
    lines.append("")

    for c in clustering:
        name   = c["name"]
        params = {k: v for k, v in c.get("best_params", {}).items()
                  if k not in ("anomalies_found", "outliers_found")}
        sil    = c.get("silhouette", 0)
        nlo    = name.lower()
        var    = nlo.replace(" ", "_").replace("-", "_")
        ps     = ", ".join([f"{k}={repr(v)}" for k, v in params.items()])
        if ps: ps += ", "

        lines.append(f"# ── {name} — Silhouette: {sil:.4f}")
        lines.append(f"# Best params: {params}")

        if   "k-means"   in nlo or "kmeans" in nlo: lines.append(f"model_{var} = KMeans({ps}random_state=42)")
        elif "dbscan"    in nlo:                     lines.append(f"model_{var} = DBSCAN({ps})")
        elif "agglomer"  in nlo or "hier" in nlo:   lines.append(f"model_{var} = AgglomerativeClustering({ps})")
        elif "gaussian"  in nlo or "gmm"  in nlo:   lines.append(f"model_{var} = GaussianMixture({ps}random_state=42)")
        elif "isolation" in nlo:                     lines.append(f"model_{var} = IsolationForest({ps}random_state=42)")
        elif "lof"       in nlo or "outlier" in nlo: lines.append(f"model_{var} = LocalOutlierFactor({ps})")

        if "isolation" in nlo or "lof" in nlo or "outlier" in nlo:
            lines.append(f"labels_{var} = np.where(model_{var}.fit_predict(X_scaled)==1, 0, 1)")
        elif "gaussian" in nlo or "gmm" in nlo:
            lines.append(f"model_{var}.fit(X_scaled)")
            lines.append(f"labels_{var} = model_{var}.predict(X_scaled)")
        else:
            lines.append(f"labels_{var} = model_{var}.fit_predict(X_scaled)")

        lines.append(f"sil_{var} = silhouette_score(X_scaled, labels_{var})")
        lines.append(f'all_results["{name}"] = {{"silhouette": sil_{var}}}')
        lines.append(f'print(f"{name}: Silhouette={{sil_{var}:.4f}}, Clusters={{len(np.unique(labels_{var}))}}")')
        lines.append(f"plt.figure(figsize=(7,5))")
        lines.append(f"plt.scatter(X_2d[:,0], X_2d[:,1], c=labels_{var}, cmap='tab10', s=15, alpha=0.7)")
        lines.append(f"plt.title('{name}  Silhouette={{sil_{var}:.4f}}')")
        lines.append(f"plt.xlabel('PCA 1'); plt.ylabel('PCA 2')")
        lines.append(f"plt.colorbar(label='Cluster'); plt.tight_layout()")
        lines.append(f"plt.savefig('{var}.png', dpi=150); plt.show()")
        lines.append("")

    lines.append('print("\\n" + "="*50)')
    lines.append('best = max(all_results, key=lambda k: all_results[k]["silhouette"])')
    lines.append('print(f"Best: {best}  Silhouette={all_results[best][\'silhouette\']:.4f}")')
    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────
def run_unsupervised(data_path: str, selected_algos: list,
                     result_path: str, n_trials: int = 20,
                     progress_callback=None) -> dict:
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return {"error": f"Could not load dataset: {str(e)}"}

    X_scaled, feature_names = preprocess_unsupervised(df)

    results = {
        "task":              "unsupervised",
        "dataset_shape":     list(df.shape),
        "feature_names":     feature_names,
        "n_trials_per_algo": n_trials,
        "clustering":        [],
        "pca_variance":      compute_pca_variance(X_scaled),
        "pca_coords":        compute_pca_2d(X_scaled),
    }

    total = len(selected_algos)

    for i, algo_name in enumerate(selected_algos):
        fn = _resolve_algo(algo_name)
        if fn is None:
            results["clustering"].append({
                "name": algo_name, "silhouette": None,
                "best_params": {}, "error": f"Unknown: {algo_name}"
            })
        else:
            try:
                res      = fn(X_scaled, n_trials)
                res["name"] = algo_name
                results["clustering"].append(res)
            except Exception as e:
                results["clustering"].append({
                    "name": algo_name, "silhouette": None,
                    "best_params": {}, "error": str(e)
                })

        if progress_callback:
            progress_callback(algo_name, i + 1, total)

    # Sort best silhouette first
    valid   = [c for c in results["clustering"] if c.get("silhouette") is not None]
    invalid = [c for c in results["clustering"] if c.get("silhouette") is None]
    valid.sort(key=lambda x: x["silhouette"], reverse=True)
    results["clustering"] = valid + invalid

    if valid:
        results["best_labels"] = valid[0].get("labels", [])
        results["best_algo"]   = valid[0]["name"]
    else:
        results["best_labels"] = []
        results["best_algo"]   = ""

    results["final_code"] = _generate_unsupervised_code(results, data_path)

    # Persist (strip large arrays)
    save_r = {k: v for k, v in results.items()
              if k not in ("pca_coords", "best_labels")}
    for c in save_r.get("clustering", []):
        c.pop("labels", None)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(save_r, f, indent=4)

    return results