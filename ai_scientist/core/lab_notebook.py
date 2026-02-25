import sqlite3
import json
import os
from datetime import datetime


def _get_db_path():
    # Import here to avoid circular import at module load
    from config import DB_PATH
    return DB_PATH


def _get_conn():
    db_path = _get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def init_db():
    """Create experiments table if it doesn't exist."""
    conn = _get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            user_prompt   TEXT    NOT NULL,
            task          TEXT,
            selected_models TEXT,
            results       TEXT,
            insight       TEXT,
            dataset_shape TEXT,
            best_model    TEXT,
            best_score    REAL
        )
    """)
    conn.commit()
    conn.close()


def save_experiment(user_prompt: str, results: dict, insight: str, selected_models: list):
    """Persist a completed experiment to the lab notebook."""
    init_db()
    conn = _get_conn()
    c = conn.cursor()

    task          = results.get("task", "unknown")
    dataset_shape = str(results.get("dataset_shape", []))
    models_json   = json.dumps(selected_models)
    results_json  = json.dumps(results)

    # Find best model
    best_model = ""
    best_score = None
    valid = [m for m in results.get("models", []) if m.get("score") is not None]
    if valid:
        if task == "classification":
            best = max(valid, key=lambda x: x["score"])
        else:
            best = min(valid, key=lambda x: x["score"])
        best_model = best["name"]
        best_score = best["score"]

    c.execute("""
        INSERT INTO experiments
            (timestamp, user_prompt, task, selected_models, results,
             insight, dataset_shape, best_model, best_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user_prompt, task, models_json, results_json,
        insight, dataset_shape, best_model, best_score
    ))

    conn.commit()
    conn.close()


def get_all_experiments() -> list:
    """Fetch all experiments, newest first."""
    init_db()
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM experiments ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    cols = ["id", "timestamp", "user_prompt", "task", "selected_models",
            "results", "insight", "dataset_shape", "best_model", "best_score"]

    experiments = []
    for row in rows:
        exp = dict(zip(cols, row))
        exp["results"]         = json.loads(exp["results"])         if exp["results"]         else {}
        exp["selected_models"] = json.loads(exp["selected_models"]) if exp["selected_models"] else []
        experiments.append(exp)

    return experiments


def get_experiment_by_id(exp_id: int):
    """Fetch a single experiment by ID."""
    init_db()
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    cols = ["id", "timestamp", "user_prompt", "task", "selected_models",
            "results", "insight", "dataset_shape", "best_model", "best_score"]
    exp = dict(zip(cols, row))
    exp["results"]         = json.loads(exp["results"])         if exp["results"]         else {}
    exp["selected_models"] = json.loads(exp["selected_models"]) if exp["selected_models"] else []
    return exp


def clear_all_experiments():
    """Wipe all experiments."""
    init_db()
    conn = _get_conn()
    conn.cursor().execute("DELETE FROM experiments")
    conn.commit()
    conn.close()
