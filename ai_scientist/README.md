# ai_scientist (Backend Pipeline)

This folder contains the AAS backend: orchestration, LLM agents, AutoML/unsupervised engines, explainability, self-improvement loop, persistence, and report generation.

## Core Modules

1. `automl_engine.py` — supervised model registry, Optuna tuning, CV scoring, ensemble, and final code generation.
2. `cluster_profiler.py` — cluster profiling and narrative summaries for unsupervised results.
3. `coder.py` — Groq-based code generator helper module.
4. `data_health.py` — dataset quality scoring and issue detection.
5. `feature_engineer.py` — LLM-driven feature suggestion and expression application.
6. `lab_notebook.py` — SQLite schema, insert/query/export, and experiment history utilities.
7. `report_generator.py` — PDF report builder with fallback text report path.
8. `researcher.py` — Groq call wrapper, model/algo decisions, hypothesis, and insight generation.
9. `self_improve.py` — round-2 improvement planning and round comparison logic.
10. `shap_explainer.py` — SHAP explainability routing and best-model explanation flow.
11. `unsupervised_engine.py` — clustering/anomaly optimization, PCA outputs, and unsupervised code generation.

## Environment Setup

Create `ai_scientist/.env` and set:

```env
GROQ_API_KEY=your_groq_key
```

You can copy from `../.env.example` first.

## Install

```bash
cd ai_scientist
pip install -r requirements.txt
```

## Run Standalone

```bash
python app.py
```

## Config (`config.py`)

`config.py` defines:
- File paths: `DATA_PATH`, `OUTPUT_DIR`, `RESULT_PATH`, `DB_PATH`, `REPORT_PATH`, `LOGS_PATH`, `GENERATED_CODE_PATH`
- LLM model names: `MODEL_RESEARCHER`, `MODEL_CODER`
- Retry limit: `MAX_RETRIES`

## Data Input

For supervised runs, place a CSV with a `target` column at:

```text
data/sample.csv
```

The Streamlit UI writes uploaded files to the same path.

## Outputs (`outputs/`)

The pipeline writes/uses:
- `report.pdf` (or text fallback report path when PDF backend is unavailable)
- `generated_script.py` (configured output code path)
- `paper_results.csv` (lab notebook export summary)
- `lab_notebook.db` (SQLite experiment history)
- `results.json` (latest serialized run results)
