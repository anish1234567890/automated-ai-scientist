# ai_scientist — Backend

This folder contains the full ML pipeline: LLM agents, AutoML engine, lab notebook, and report generator.

## Setup

```bash
cd ai_scientist
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## API Key

Copy `.env.example` from the repo root to this folder:

```bash
copy ..\.env.example .env        # Windows
# cp ../.env.example .env        # Mac/Linux
```

Add your Groq API key inside `.env`:
```
GROQ_API_KEY=your_key_here
```

Get a free key at: https://console.groq.com

## Dataset

Place your CSV as `data/sample.csv` — or upload it via the Streamlit UI.

Your CSV **must have a column named `target`** (the column to predict).

## Run pipeline directly (no UI)

```bash
python app.py
```

## Folder contents

| File/Folder | Purpose |
|---|---|
| `app.py` | Main orchestrator — runs the full pipeline |
| `config.py` | All file paths and settings |
| `core/automl_engine.py` | 21 ML models + Optuna hyperparameter tuning |
| `core/researcher.py` | LLM model selector + scientific insight generator |
| `core/lab_notebook.py` | SQLite persistent experiment history |
| `core/report_generator.py` | PDF report export |
| `data/` | Put your CSV datasets here |
| `outputs/` | Generated files: results.json, report.pdf, lab_notebook.db |