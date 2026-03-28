import sqlite3
import json
import os
from datetime import datetime


# ================= SAFE JSON LOADER =================
def _safe_json_load(value, default):
    try:
        return json.loads(value) if value else default
    except Exception:
        return default


def _get_db_path():
    from config import DB_PATH
    return DB_PATH


def _get_conn():
    db_path = _get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def init_db():
    conn = _get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            user_prompt     TEXT    NOT NULL,
            mode            TEXT    DEFAULT 'supervised',
            task            TEXT,
            selected_models TEXT,
            results         TEXT,
            insight         TEXT,
            dataset_shape   TEXT,
            best_model      TEXT,
            best_score      REAL
        )
    """)

    try:
        c.execute("ALTER TABLE experiments ADD COLUMN mode TEXT DEFAULT 'supervised'")
        conn.commit()
    except Exception:
        pass

    conn.commit()
    conn.close()


def save_experiment(user_prompt: str, results: dict, insight: str, selected_models: list, mode: str = 'supervised'):
    init_db()
    conn = _get_conn()
    c = conn.cursor()

    dataset_shape = str(results.get("dataset_shape", []))
    models_json   = json.dumps(selected_models)

    # remove heavy data
    save_results = {k: v for k, v in results.items()
                    if k not in ("best_labels", "pca_coords")}

    for c2 in save_results.get("clustering", []):
        c2.pop("labels", None)

    results_json = json.dumps(save_results)

    best_model = ""
    best_score = None
    task       = results.get("task", "unknown")

    if mode == "supervised":
        valid = [m for m in results.get("models", []) if m.get("score") is not None]
        if valid:
            if task == "classification":
                best = max(valid, key=lambda x: x["score"])
            else:
                best = min(valid, key=lambda x: x["score"])
            best_model = best["name"]
            best_score = best["score"]
    else:
        task = "unsupervised"
        valid = [c2 for c2 in results.get("clustering", []) if c2.get("silhouette") is not None]
        if valid:
            best = max(valid, key=lambda x: x["silhouette"])
            best_model = best["name"]
            best_score = best["silhouette"]

    c.execute("""
        INSERT INTO experiments
            (timestamp, user_prompt, mode, task, selected_models, results,
             insight, dataset_shape, best_model, best_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user_prompt, mode, task, models_json, results_json,
        insight, dataset_shape, best_model, best_score
    ))

    conn.commit()
    conn.close()


def get_all_experiments() -> list:
    init_db()
    conn = _get_conn()
    c = conn.cursor()

    c.execute("SELECT * FROM experiments ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    cols = ["id", "timestamp", "user_prompt", "mode", "task", "selected_models",
            "results", "insight", "dataset_shape", "best_model", "best_score"]

    experiments = []

    for row in rows:
        exp = dict(zip(cols, row))

        # 🔥 FIXED SAFE LOADING
        exp["results"] = _safe_json_load(exp.get("results"), {})
        exp["selected_models"] = _safe_json_load(exp.get("selected_models"), [])

        experiments.append(exp)

    return experiments


def get_experiment_by_id(exp_id: int):
    init_db()
    conn = _get_conn()
    c = conn.cursor()

    c.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    cols = ["id", "timestamp", "user_prompt", "mode", "task", "selected_models",
            "results", "insight", "dataset_shape", "best_model", "best_score"]

    exp = dict(zip(cols, row))

    # 🔥 FIXED SAFE LOADING
    exp["results"] = _safe_json_load(exp.get("results"), {})
    exp["selected_models"] = _safe_json_load(exp.get("selected_models"), [])

    return exp


def clear_all_experiments():
    init_db()
    conn = _get_conn()
    conn.cursor().execute("DELETE FROM experiments")
    conn.commit()
    conn.close()