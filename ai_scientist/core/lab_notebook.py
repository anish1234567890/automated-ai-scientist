import sqlite3
import json
import os
import csv
from datetime import datetime


def _get_db_path():
    from config import DB_PATH
    return DB_PATH


def _get_conn():
    db_path = _get_db_path()
    _dir = os.path.dirname(db_path)
    if _dir:
        os.makedirs(_dir, exist_ok=True)
    return sqlite3.connect(db_path)


def _safe_json_load(value, default):
    try:
        return json.loads(value) if value else default
    except Exception:
        return default


def _format_score(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "-"


def _format_delta(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):+.2f}"
    except Exception:
        return "-"


def _format_time_minutes(seconds):
    if seconds is None:
        return "-"
    try:
        return f"{float(seconds) / 60.0:.1f}m"
    except Exception:
        return "-"


def _short_shap_name(name):
    if not name:
        return "-"
    name = str(name).strip().lower().replace("explainer", "")
    if name == "tree":
        return "Tree"
    if name == "linear":
        return "Linear"
    if name == "kernel":
        return "Kernel"
    return name.title() or "-"


def _pad(text, width, align="left"):
    text = "" if text is None else str(text)
    if len(text) > width:
        text = text[: width - 1] + "…"
    if align == "right":
        return text.rjust(width)
    return text.ljust(width)


def init_db():
    conn = _get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            user_prompt         TEXT    NOT NULL,
            mode                TEXT    DEFAULT 'supervised',
            task                TEXT,
            selected_models     TEXT,
            results             TEXT,
            insight             TEXT,
            dataset_shape       TEXT,
            best_model          TEXT,
            best_score          REAL,
            round1_score        REAL,
            round1_model        TEXT,
            round2_score        REAL,
            round2_model        TEXT,
            delta               REAL,
            shap_explainer      TEXT,
            ensemble_score      REAL,
            health_grade        TEXT,
            experiment_time_sec REAL,
            dataset_name        TEXT,
            features_added      INTEGER
        )
    """)

    alter_statements = [
        "ALTER TABLE experiments ADD COLUMN mode TEXT DEFAULT 'supervised'",
        "ALTER TABLE experiments ADD COLUMN round1_score REAL",
        "ALTER TABLE experiments ADD COLUMN round1_model TEXT",
        "ALTER TABLE experiments ADD COLUMN round2_score REAL",
        "ALTER TABLE experiments ADD COLUMN round2_model TEXT",
        "ALTER TABLE experiments ADD COLUMN delta REAL",
        "ALTER TABLE experiments ADD COLUMN shap_explainer TEXT",
        "ALTER TABLE experiments ADD COLUMN ensemble_score REAL",
        "ALTER TABLE experiments ADD COLUMN health_grade TEXT",
        "ALTER TABLE experiments ADD COLUMN experiment_time_sec REAL",
        "ALTER TABLE experiments ADD COLUMN dataset_name TEXT",
        "ALTER TABLE experiments ADD COLUMN features_added INTEGER",
    ]

    for stmt in alter_statements:
        try:
            c.execute(stmt)
            conn.commit()
        except Exception:
            pass

    conn.commit()
    conn.close()


def save_experiment(
    user_prompt,
    results,
    insight,
    selected_models,
    mode='supervised',
    round1_score=None,
    round1_model=None,
    round2_score=None,
    round2_model=None,
    delta=None,
    shap_explainer=None,
    ensemble_score=None,
    health_grade=None,
    experiment_time_sec=None,
    dataset_name=None,
    features_added=None,
):
    init_db()
    conn = _get_conn()
    c = conn.cursor()

    dataset_shape = str(results.get("dataset_shape", []))
    models_json = json.dumps(selected_models)

    save_results = {k: v for k, v in results.items() if k not in ("best_labels", "pca_coords")}
    for c2 in save_results.get("clustering", []):
        c2.pop("labels", None)
    results_json = json.dumps(save_results)

    best_model = ""
    best_score = None
    task = results.get("task", "unknown")

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

    if round1_score is None:
        round1_score = best_score
    if round1_model is None:
        round1_model = best_model
    if ensemble_score is None:
        ensemble_score = (results.get("ensemble") or {}).get("cv_score")

    if shap_explainer is None:
        shap_block = results.get("shap", {}) or {}
        expl = shap_block.get("explainer_type")
    else:
        expl = shap_explainer

    # Normalise raw lowercase values ('tree','linear','kernel') to full names
    _expl_map = {"tree": "TreeExplainer", "linear": "LinearExplainer", "kernel": "KernelExplainer"}
    shap_explainer = _expl_map.get(str(expl).lower(), expl) if expl else None

    if delta is None and round1_score is not None and round2_score is not None:
        try:
            delta = float(round2_score) - float(round1_score)
        except Exception:
            delta = None

    c.execute("""
        INSERT INTO experiments (
            timestamp, user_prompt, mode, task, selected_models, results,
            insight, dataset_shape, best_model, best_score,
            round1_score, round1_model, round2_score, round2_model, delta,
            shap_explainer, ensemble_score, health_grade, experiment_time_sec,
            dataset_name, features_added
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user_prompt,
        mode,
        task,
        models_json,
        results_json,
        insight,
        dataset_shape,
        best_model,
        best_score,
        round1_score,
        round1_model,
        round2_score,
        round2_model,
        delta,
        shap_explainer,
        ensemble_score,
        health_grade,
        experiment_time_sec,
        dataset_name,
        features_added,
    ))

    row_id = c.lastrowid
    conn.commit()
    conn.close()
    print(f"Saved experiment #{row_id} to lab notebook")


def get_all_experiments():
    init_db()
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM experiments ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    cols = [
        "id", "timestamp", "user_prompt", "mode", "task", "selected_models",
        "results", "insight", "dataset_shape", "best_model", "best_score",
        "round1_score", "round1_model", "round2_score", "round2_model", "delta",
        "shap_explainer", "ensemble_score", "health_grade", "experiment_time_sec",
        "dataset_name", "features_added"
    ]

    experiments = []
    for row in rows:
        exp = dict(zip(cols, row))
        exp["results"] = _safe_json_load(exp.get("results"), {})
        exp["selected_models"] = _safe_json_load(exp.get("selected_models"), [])
        experiments.append(exp)

    return experiments


def get_experiment_by_id(exp_id):
    init_db()
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    cols = [
        "id", "timestamp", "user_prompt", "mode", "task", "selected_models",
        "results", "insight", "dataset_shape", "best_model", "best_score",
        "round1_score", "round1_model", "round2_score", "round2_model", "delta",
        "shap_explainer", "ensemble_score", "health_grade", "experiment_time_sec",
        "dataset_name", "features_added"
    ]

    exp = dict(zip(cols, row))
    exp["results"] = _safe_json_load(exp.get("results"), {})
    exp["selected_models"] = _safe_json_load(exp.get("selected_models"), [])
    return exp


def export_to_csv(output_path=None):
    if output_path is None:
        try:
            from config import OUTPUT_DIR
            output_path = os.path.join(OUTPUT_DIR, "paper_results.csv")
        except Exception:
            output_path = os.path.join("outputs", "paper_results.csv")
    init_db()
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    cols = [
        "id", "timestamp", "dataset_name", "task", "round1_model", "round1_score",
        "round2_model", "round2_score", "delta", "shap_explainer", "ensemble_score",
        "health_grade", "experiment_time_sec", "features_added", "best_score"
    ]

    query = f"SELECT {', '.join(cols)} FROM experiments ORDER BY id DESC"
    rows = c.execute(query).fetchall()
    conn.close()

    _out_dir = os.path.dirname(output_path)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in cols})

    print("Saved to paper_results.csv")


def print_paper_summary():
    init_db()
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    rows = c.execute("""
        SELECT
            dataset_name, round1_score, round2_score, delta,
            shap_explainer, experiment_time_sec, health_grade
        FROM experiments
        ORDER BY id DESC
    """).fetchall()
    conn.close()

    top =    "╔══════════════════════════════════════════════════════════════╗"
    title =  "║         AAS PAPER RESULTS SUMMARY                           ║"
    mid1 =   "╠══════════╦═══════╦════════╦════════╦════════╦══════╦════════╣"
    hdr =    "║ Dataset  ║ R1    ║ R2     ║ Delta  ║ SHAP   ║ Time ║ Grade  ║"
    mid2 =   "╠══════════╬═══════╬════════╬════════╬════════╬══════╬════════╣"
    mid3 =   "╠══════════╬═══════╬════════╬════════╬════════╬══════╬════════╣"
    bottom = "╚══════════╩═══════╩════════╩════════╩════════╩══════╩════════╝"

    print(top)
    print(title)
    print(mid1)
    print(hdr)
    print(mid2)

    r1_vals = []
    r2_vals = []
    delta_vals = []
    time_vals = []

    for row in rows:
        dataset = row["dataset_name"] or "N/A"
        r1 = row["round1_score"]
        r2 = row["round2_score"]
        d = row["delta"]
        shap = _short_shap_name(row["shap_explainer"])
        t = row["experiment_time_sec"]
        grade = row["health_grade"] or "-"

        if r1 is not None:
            r1_vals.append(float(r1))
        if r2 is not None:
            r2_vals.append(float(r2))
        if d is not None:
            delta_vals.append(float(d))
        if t is not None:
            time_vals.append(float(t) / 60.0)

        line = (
            f"║ {_pad(dataset, 8)} "
            f"║ {_pad(_format_score(r1), 5, 'right')} "
            f"║ {_pad(_format_score(r2), 6, 'right')} "
            f"║ {_pad(_format_delta(d), 6, 'right')} "
            f"║ {_pad(shap, 6)} "
            f"║ {_pad(_format_time_minutes(t), 4, 'right')} "
            f"║ {_pad(grade, 6)} ║"
        )
        print(line)

    avg_r1 = f"{sum(r1_vals) / len(r1_vals):.2f}" if r1_vals else "-"
    avg_r2 = f"{sum(r2_vals) / len(r2_vals):.2f}" if r2_vals else "-"
    avg_delta = f"{sum(delta_vals) / len(delta_vals):+.2f}" if delta_vals else "-"
    avg_time = f"{sum(time_vals) / len(time_vals):.1f}" if time_vals else "-"

    print(mid3)
    avg_line = (
        f"║ {_pad('AVERAGE', 8)} "
        f"║ {_pad(avg_r1, 5, 'right')} "
        f"║ {_pad(avg_r2, 6, 'right')} "
        f"║ {_pad(avg_delta, 6, 'right')} "
        f"║ {_pad('-', 6)} "
        f"║ {_pad(avg_time, 4, 'right')} "
        f"║ {_pad('-', 6)} ║"
    )
    print(avg_line)
    print(bottom)


def clear_all_experiments():
    init_db()
    conn = _get_conn()
    conn.cursor().execute("DELETE FROM experiments")
    conn.commit()
    conn.close()
    print("Cleared all experiments from lab notebook")