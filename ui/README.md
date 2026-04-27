# UI (Streamlit Frontend)

This folder contains the Streamlit web interface for Automated AI Scientist (AAS). The UI handles dataset upload, pipeline execution, real-time progress, result visualization, report/code downloads, and lab notebook browsing.

## Run

```bash
cd ui
streamlit run streamlit_app.py
```

## Requirements

- streamlit
- pandas
- numpy
- altair

## Tabs and Pages

### Page: 🚀 Run Experiment

#### Supervised mode tabs
1. 🏆 Leaderboard
2. 🔄 Self-Improve
3. 🔧 Features
4. ⚙️ Parameters
5. 🔍 SHAP
6. 🔬 AI Insights
7. 💻 Final Code
8. 📄 Report

#### Unsupervised mode tabs
1. 🔵 Clusters
2. 🔴 Anomalies
3. 📋 Profiles
4. 🔬 AI Insights
5. 💻 Final Code
6. 📄 Report

### Page: 📓 Lab Notebook

- Shows saved supervised and unsupervised experiments from SQLite.
- Displays per-experiment metadata, metrics, insights, and download buttons.

## Connection to Backend (`ai_scientist/app.py`)

- `streamlit_app.py` adds `ai_scientist` to `sys.path` and imports:
  - `run_ai_scientist` from `ai_scientist/app.py`
  - `get_all_experiments`, `clear_all_experiments` from `core/lab_notebook.py`
- Clicking **Run AI Scientist v3.0** calls `run_ai_scientist(...)` and renders the returned dictionary (`results`, `results_r2`, `insight`, `health`, `shap_result`, `report_path`, logs, etc.).
