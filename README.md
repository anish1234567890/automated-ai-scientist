# 🧪 Automated AI Scientist — v2.0

> **A hybrid AutoML system where an LLM interprets natural language instructions and an Optuna-based engine automatically trains, tunes, and compares 21 machine learning models.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)
[![Optuna](https://img.shields.io/badge/Tuning-Optuna-blue)](https://optuna.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.3%2070B-orange)](https://groq.com)

---

## 🎯 What Is This?

Traditional AutoML tools require you to pick models and tune parameters manually. This system lets you just **type what you want**:

```
try random forest, xgboost and ridge regression on this dataset
```

And the system:
- **Understands** your intent using an LLM (Llama 3.3 70B)
- **Selects** the right models automatically
- **Tunes** hyperparameters using Optuna (25 trials per model, TPE sampler)
- **Ranks** all models in a leaderboard
- **Explains** the results with AI-generated scientific insights
- **Exports** a downloadable PDF report + runnable Python code

---

## 🏗️ Architecture

```
User (natural language prompt)
            │
            ▼
  ┌─────────────────────┐
  │  LLM Researcher     │  ← Llama 3.3 70B (Groq API)
  │  (decides models)   │     Maps "l1 regression" → Lasso
  └────────┬────────────┘     Maps "gbm" → Gradient Boosting
           │                  Maps "knn" → K-Nearest Neighbors
           ▼
  ┌─────────────────────┐
  │  AutoML Engine      │  ← Optuna TPE Sampler
  │  (trains & tunes)   │     25 trials per model
  └────────┬────────────┘     Detects task type automatically
           │
           ▼
  ┌─────────────────────┐
  │  LLM Insight Agent  │  ← Llama 3.3 70B (Groq API)
  │  (scientific        │     Explains why best model won
  │   analysis)         │     Flags overfitting/underfitting
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────┐
  │  Streamlit Dashboard                            │
  │  • Leaderboard  • Hyperparameter breakdown      │
  │  • AI Insights  • Downloadable Python code      │
  │  • PDF Report   • Persistent Lab Notebook       │
  └─────────────────────────────────────────────────┘
```

---

## 🤖 Supported Models (21 Total)

| Category | Models |
|---|---|
| **Boosting** | XGBoost, LightGBM, CatBoost, Gradient Boosting, AdaBoost |
| **Ensemble** | Random Forest, Extra Trees, Bagging |
| **Linear** | Linear Regression, Ridge, Lasso, Elastic Net, Logistic Regression, Bayesian Ridge, Huber, SGD |
| **Other** | SVM, KNN, Decision Tree, Naive Bayes, LDA |

All models are tuned with **Optuna TPE** — each with their own search space (e.g. XGBoost tunes 7 params, Ridge tunes `alpha` on log scale, KNN tunes `n_neighbors`, `weights`, and `metric`).

---

## 📁 Project Structure

```
automated-ai-scientist/
│
├── ai_scientist/                  ← Backend (ML engine + agents)
│   ├── core/
│   │   ├── automl_engine.py       ← 21 models + Optuna tuning + code generator
│   │   ├── researcher.py          ← LLM model selector + insight generator (with retry)
│   │   ├── lab_notebook.py        ← SQLite experiment history
│   │   └── report_generator.py   ← PDF export (fpdf2)
│   ├── data/
│   │   └── sample.csv             ← Upload your CSV via UI
│   ├── outputs/
│   │   ├── results.json           ← Latest results
│   │   ├── lab_notebook.db        ← All past experiments (SQLite)
│   │   └── report.pdf             ← Latest PDF report
│   ├── app.py                     ← Main pipeline orchestrator
│   ├── config.py                  ← Paths + settings
│   ├── requirements.txt
│   └── .env                       ← Your API keys (not committed)
│
├── ui/
│   └── streamlit_app.py           ← 5-tab dashboard + Lab Notebook page
│
├── .env.example                   ← Copy to ai_scientist/.env
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/anish1234567890/automated-ai-scientist.git
cd automated-ai-scientist
git checkout anish
```

### 2. Create virtual environment (recommended)

```bash
cd ai_scientist
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Optional models (LightGBM + CatBoost):
```bash
pip install lightgbm catboost
```

### 4. Set up API key

```bash
# From the repo root
copy .env.example ai_scientist\.env     # Windows
# cp .env.example ai_scientist/.env    # Mac/Linux
```

Open `ai_scientist/.env` and add your key:
```
GROQ_API_KEY=your_key_here
```

Get a **free** Groq API key at: https://console.groq.com

### 5. Run the dashboard

```bash
cd ui
streamlit run streamlit_app.py
```

---

## 📋 Dataset Requirements

Your CSV must have a column named **`target`** — the column to predict.

```csv
age, income, education, target
25, 50000, 16, 1
32, 75000, 18, 0
```

The system **auto-detects** classification vs regression from the target column. No configuration needed.

---

## 🖥️ Dashboard Tabs

| Tab | Description |
|---|---|
| 🏆 **Leaderboard** | Ranked models with scores + bar chart comparison |
| ⚙️ **Parameters** | Full Optuna best hyperparameters for every model |
| 🔬 **AI Insights** | LLM-generated scientific analysis of results |
| 💻 **Final Code** | Fully runnable Python with best params — downloadable |
| 📄 **Report** | Downloadable PDF report + full agent logs |
| 📓 **Lab Notebook** | All past experiments with code download per run |

---

## 💬 Example Prompts

```
try all boosting models
compare random forest and xgboost for accuracy
use l1 and l2 regression
try everything and find the best model
use fast models for classification
compare knn, decision tree and naive bayes
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq API |
| Hyperparameter Tuning | Optuna (TPE Sampler) |
| ML Models | Scikit-learn, XGBoost, LightGBM, CatBoost |
| UI | Streamlit |
| Database | SQLite (Python standard library) |
| PDF Reports | fpdf2 |
| Language | Python 3.10+ |

---

## 💼 For Interviews / Viva

> *"I built a hybrid AutoML system that combines LLM-based model selection with an Optuna-powered hyperparameter tuning engine. The user provides instructions in natural language — the LLM decides which of 21 ML models to try, Optuna tunes each model's hyperparameters over 25 trials using TPE sampling, a second LLM agent generates scientific insights on the results, and the system outputs a ranked leaderboard, downloadable Python code with the best parameters baked in, and a PDF research report. All experiments are stored in a SQLite lab notebook for reproducibility."*

**Key innovation:** Unlike standard AutoML tools, this system understands natural language intent. The user doesn't need to know model names or hyperparameter ranges — they describe their goal and the system figures out the rest.

---

## 🔮 Future Scope

- Deep learning support (PyTorch / TensorFlow)
- Cross-validation instead of single train/test split
- SHAP feature importance visualization
- Cloud deployment (Streamlit Cloud / HuggingFace Spaces)
- Ensemble of top-k models
- Multi-dataset experiment workspace

---

## 🙏 Acknowledgments

- Inspired by [Sakana AI's AI Scientist](https://github.com/SakanaAI/AI-Scientist)
- Inspired by [Google Research's AI Co-Scientist](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/)
- Built with [Groq](https://groq.com) · [Optuna](https://optuna.org) · [Streamlit](https://streamlit.io)

---

**Built by Anish** | [GitHub](https://github.com/anish1234567890/automated-ai-scientist)