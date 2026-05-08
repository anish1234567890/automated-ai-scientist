<!-- v4.0 -->

# 🖥️ UI Reference (`ui/streamlit_app.py`)

This Streamlit app is the frontend for the backend pipeline in `ai_scientist`. It uploads data, runs experiments, streams progress, renders supervised/unsupervised results, and reads/wipes notebook history.

## 🚀 Run

```bash
pip install -r ai_scientist/requirements.txt
streamlit run ui\streamlit_app.py
```

## 📂 Dataset upload behavior

| Behavior | Code path |
|---|---|
| Upload widget | `st.file_uploader("Upload CSV", type=["csv"])` |
| Save location | `ai_scientist\data\sample.csv` |
| Supervised readiness check | if uploaded frame contains `target` column |
| Missing `target` message | UI shows unsupervised info; backend mode routing still happens in `app.py` |

## 🔁 How supervised vs unsupervised mode is selected

There is no manual mode toggle in the UI. Mode is chosen by backend logic:

1. `run_ai_scientist()` calls `should_run_unsupervised(df, user_prompt)`.
2. It returns **unsupervised** if prompt includes keywords like `cluster`, `anomaly`, `outlier`, etc., or if dataset has no `target`.
3. Otherwise it runs **supervised**.

## 📊 Progress bars (trial-level + model-level + overall)

| Level | What updates it | How it is calculated |
|---|---|---|
| Trial-level (inside current model) | Backend emits `trial X/Y` messages from Optuna callback | `current_trial / current_trial_total` (capped at `0.99` until model finalizes) |
| Model-level panel | Backend emits `Tuning [i/total]: model_name` | Shows current model name, index, trial status, and best-so-far score |
| Overall progress | Model finalization events | `completed / total` for current round |
| Round handling | Backend stage text `AutoML Round 2` | UI switches to Round 2 header and resets overall round stats |

## 🧭 Pages and tabs

### 🚀 Run Experiment page

#### Supervised tabs

| Tab | What it shows |
|---|---|
| 🏆 Leaderboard | Ranked models, CV metric table, ensemble summary, bar chart |
| 🔄 Self-Improve | Round 1 vs Round 2 deltas, LLM improvement plan, applied changes, round-2 leaderboard |
| 🔧 Features | LLM-generated engineered features with expression/rationale/status |
| ⚙️ Parameters | Per-model best params, trials, failures, ensemble details |
| 🔍 SHAP | Best-model SHAP top features, explainer type, SHAP chart/table |
| 🔬 AI Insights | Hypothesis and final scientific narrative insight |
| 💻 Final Code | Generated runnable Python code + download button |
| 📄 Report | Report download (`.pdf` or fallback text extension) + full agent logs |

#### Unsupervised tabs

| Tab | What it shows |
|---|---|
| 🔵 Clusters | Clustering leaderboard, PCA scatter, per-algorithm params/metrics |
| 🔴 Anomalies | Isolation Forest / LOF anomaly counts and optional PCA anomaly plot |
| 📋 Profiles | Cluster defining features, categorical modes, feature summary table |
| 🔬 AI Insights | Unsupervised scientific analysis text |
| 💻 Final Code | Generated unsupervised Python code + download |
| 📄 Report | Report download + logs |

### 📓 Lab Notebook page

| Section | What it does |
|---|---|
| Metrics row | Total experiments, supervised count, unsupervised count, classification count |
| Experiment expanders | Prompt, mode/task, dataset shape, selected algorithms/models, result tables, insight |
| Code download | Per-experiment generated code download (`ml_code_exp_<id>.py` or `unsupervised_code_exp_<id>.py`) |
| Clear button | Deletes all notebook records via `clear_all_experiments()` |

## 🔌 Backend integration

| Import | Usage |
|---|---|
| `from app import run_ai_scientist` | Executes full experiment pipeline |
| `from core.lab_notebook import get_all_experiments, clear_all_experiments` | Notebook page listing and clear action |
| `from config import DATA_PATH` | Upload/run precondition checks |
