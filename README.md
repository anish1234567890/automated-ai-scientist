# Automated AI Scientist (AAS)

Automated AI Scientist (AAS) is a hybrid ML system that takes natural-language experiment goals, routes them through an LLM researcher agent, runs supervised or unsupervised optimization pipelines, generates explainability and self-improvement rounds, and produces persistent experiment records, downloadable code, and reports through a Streamlit interface.

## Features

### 1. LLM Researcher Agent
- Interprets user prompts with Groq-hosted **Llama 3.3 70B**.
- Selects supervised models or unsupervised algorithms from supported registries.
- Generates pre-experiment hypotheses and post-experiment scientific insights.

### 2. Dataset Processing Module
- Loads CSV input from `ai_scientist/data/sample.csv`.
- Preprocesses categorical and missing values for both supervised and unsupervised flows.
- Runs dataset health checks (missingness, duplicates, imbalance, outliers, correlation, size warnings).

### 3. AutoML Engine
- Supports 21 supervised models with Optuna hyperparameter tuning.
- Uses 5-fold CV pipelines with polynomial features + feature selection.
- Builds top-3 voting ensembles and generates runnable final Python code.

### 4. Explainability & Self-Improvement Module
- Computes SHAP explanations for the best supervised model.
- Builds an LLM-driven round-2 improvement plan.
- Applies round-2 model/trial/feature-drop strategy and compares deltas vs round 1.

### 5. Persistence & Reporting Layer
- Saves experiments to SQLite lab notebook (`lab_notebook.db`).
- Exports paper-style summary CSV (`paper_results.csv`).
- Generates PDF reports via `fpdf2` (with text fallback if unavailable).

## Tech Stack

- Python 3.10+
- Streamlit
- Groq API
- Llama 3.3 70B (`llama-3.3-70b-versatile`)
- Optuna
- scikit-learn
- XGBoost
- LightGBM
- SHAP
- SQLite
- fpdf2

## Quick Start

```bash
git clone https://github.com/anish1234567890/automated-ai-scientist.git
cd automated-ai-scientist
cd ai_scientist
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy ..\.env.example .env
```

Set your key in `ai_scientist/.env`:

```env
GROQ_API_KEY=your_groq_key
```

Run the UI:

```bash
cd ..\ui
streamlit run streamlit_app.py
```

## Project Structure

```text
automated-ai-scientist/
├── AUDIT_REPORT.md
├── README.md
├── syntax_check.py
├── ai_scientist/
│   ├── .env
│   ├── .env.example
│   ├── README.md
│   ├── app.py
│   ├── config.py
│   ├── requirements.txt
│   ├── test.py
│   ├── data/
│   ├── outputs/
│   ├── test_outputs/
│   └── core/
│       ├── __init__.py
│       ├── automl_engine.py
│       ├── cluster_profiler.py
│       ├── coder.py
│       ├── data_health.py
│       ├── feature_engineer.py
│       ├── lab_notebook.py
│       ├── report_generator.py
│       ├── researcher.py
│       ├── self_improve.py
│       ├── shap_explainer.py
│       └── unsupervised_engine.py
├── ui/
│   ├── README.md
│   └── streamlit_app.py
└── venv/
```

## Sub-READMEs

- UI details: [`ui/README.md`](ui/README.md)
- Backend/core details: [`ai_scientist/README.md`](ai_scientist/README.md)
