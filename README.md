<!-- v4.0 -->

# 🧪 Automated AI Scientist (AAS)

Automated AI Scientist is a natural-language-driven ML system that can run either a supervised AutoML pipeline or an unsupervised clustering/anomaly pipeline from the same prompt, then generate hypothesis, SHAP explanations, round-2 self-improvement, insights, notebook records, downloadable code, and a report.

What makes it different: it is not a single-pass trainer. In supervised mode it explicitly performs a scientist-style loop (**Round 1 → analysis → LLM improvement plan → Round 2 → delta comparison**).

## 🚀 What it does

| Capability | What is implemented in code |
|---|---|
| Prompt-to-model routing | `core.researcher.decide_models()` and `decide_unsupervised_algos()` choose models/algorithms from fixed allowed lists |
| Automatic mode selection | `core.unsupervised_engine.should_run_unsupervised()` uses prompt keywords or missing `target` column |
| Dataset health audit | `core.data_health.run_health_check()` scores data (0–100), grade (A–D), issues, summary, stats |
| Supervised AutoML | `core.automl_engine.run_automl()` runs Optuna tuning per model + CV scoring + top-3 voting ensemble |
| Unsupervised optimization | `core.unsupervised_engine.run_unsupervised()` tunes clustering/anomaly algorithms and computes silhouette/DB/CH metrics |
| LLM feature engineering | `core.feature_engineer.run_feature_engineering()` proposes/evaluates feature expressions and enriches dataset |
| SHAP explainability | `core.shap_explainer.run_shap_for_best_model()` explains best supervised model |
| Self-improvement loop | `core.self_improve` builds round-2 plan/config and compares round-1 vs round-2 |
| Cluster interpretation | `core.cluster_profiler.profile_clusters()` + `generate_cluster_narrative()` |
| Persistence | `core.lab_notebook.save_experiment()` stores experiments in SQLite |
| Reports | `core.report_generator.generate_pdf_report()` writes PDF (or `.txt` fallback if `fpdf2` unavailable) |
| UI | `ui/streamlit_app.py` runs, tracks progress, and renders all tabs/downloads |

## 🧠 Full supervised model list (from `MODEL_REGISTRY`)

| Model name |
|---|
| Random Forest |
| XGBoost |
| LightGBM |
| CatBoost |
| Gradient Boosting |
| AdaBoost |
| Extra Trees |
| Bagging |
| Decision Tree |
| KNN |
| K-Nearest Neighbors (alias mapped to KNN family) |
| SVM |
| Logistic Regression |
| Linear Regression |
| Ridge Regression |
| Lasso Regression |
| Elastic Net |
| SGD |
| Bayesian Ridge |
| Huber |
| Naive Bayes |
| LDA |

## 🏗️ Architecture (actual call chain from `ai_scientist/app.py`)

```text
ui/streamlit_app.py
  └─ run_ai_scientist(user_prompt, progress_callback, n_trials, enable_feature_eng, enable_self_improve)
      ├─ Load CSV from config.DATA_PATH
      ├─ run_health_check(df)
      ├─ should_run_unsupervised(df, user_prompt)?
      │
      ├─ YES (unsupervised):
      │   ├─ decide_unsupervised_algos(user_prompt)
      │   ├─ run_unsupervised(DATA_PATH, selected_algos, RESULT_PATH, n_trials)
      │   ├─ profile_clusters(...) + generate_cluster_narrative(...)
      │   ├─ generate_unsupervised_insight(...)
      │   ├─ save_experiment(..., mode="unsupervised")
      │   └─ generate_pdf_report(..., mode="unsupervised")
      │
      └─ NO (supervised):
          ├─ decide_models(user_prompt, df)
          ├─ generate_hypothesis(...)
          ├─ optional run_feature_engineering(...)
          ├─ run_automl(train_path, selected_models, RESULT_PATH, n_trials)   [Round 1]
          ├─ run_shap_for_best_model(...)
          ├─ optional _call_llm_for_improvement_plan(...)
          │   ├─ build_improved_experiment(...)
          │   ├─ optional SHAP-based feature dropping
          │   ├─ run_automl(...Round 2 config...)                             [Round 2]
          │   └─ compare_rounds(round1, round2)
          ├─ generate_insight(...)
          ├─ save_experiment(..., mode="supervised")
          └─ generate_pdf_report(..., mode="supervised")
```

## 💬 Example prompts that work with the researcher agent

```text
Try random forest and xgboost
Compare all boosting models
Use lasso regression, ridge regression, and elastic net
Find clusters in this data
Detect anomalies with isolation forest and local outlier factor
Try everything for clustering
```

## ⚙️ Quick start (global Python install)

1. Clone:
```bash
git clone https://github.com/anish1234567890/automated-ai-scientist.git
cd automated-ai-scientist
```

2. Install dependencies:
```bash
pip install -r ai_scientist/requirements.txt
```

3. Set API key in `ai_scientist\.env`:
```env
GROQ_API_KEY=your_groq_key
```

4. Run Streamlit UI:
```bash
streamlit run ui\streamlit_app.py
```

## 🧰 Tech stack (from requirements + imports)

| Category | Libraries / tools used in code |
|---|---|
| UI | Streamlit, Altair (optional plotting path in UI) |
| LLM | Groq API, python-dotenv, model `llama-3.3-70b-versatile` |
| ML core | scikit-learn, Optuna |
| Boosting libraries | XGBoost, LightGBM, CatBoost |
| Explainability | SHAP |
| Data | pandas, numpy |
| Reporting | fpdf2 |
| Storage | SQLite (`sqlite3`) |
| Also listed in requirements | tpot, h2o, pycaret |

## 📁 Project folder structure (current workspace files)

```text
automated-ai-scientist/
├── .gitignore
├── README.md
├── ai_scientist/
│   ├── .env
│   ├── .env.example
│   ├── README.md
│   ├── app.py
│   ├── config.py
│   ├── requirements.txt
│   ├── test.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── automl_engine.py
│   │   ├── cluster_profiler.py
│   │   ├── coder.py
│   │   ├── data_health.py
│   │   ├── feature_engineer.py
│   │   ├── lab_notebook.py
│   │   ├── report_generator.py
│   │   ├── researcher.py
│   │   ├── self_improve.py
│   │   ├── shap_explainer.py
│   │   └── unsupervised_engine.py
│   ├── data/
│   │   ├── adult_income.csv
│   │   ├── breast_cancer.csv
│   │   ├── california_housing.csv
│   │   ├── diabetes.csv
│   │   ├── iris.csv
│   │   ├── sample.csv
│   │   ├── sample_enriched.csv
│   │   ├── sample_enriched_r2.csv
│   │   ├── sample_r2.csv
│   │   └── wine_quality.csv
│   ├── outputs/
│   │   ├── generated_script.py
│   │   ├── lab_notebook.db
│   │   ├── logs.txt
│   │   ├── report.pdf
│   │   └── results.json
│   ├── test_outputs/
│   │   └── test.db
│   └── catboost_info/
├── me/
│   ├── AUDIT_REPORT.md
│   ├── CHANGES_SINCE_LAST_PUSH.txt
│   ├── PROJECT_COMPLETE_ANALYSIS.md
│   ├── new_audit.md
│   └── s -ExecutionPolicy RemoteSigned) ; (& cUsersAnishOneDriveDesktopmajor projectautomated-ai-scientistvenvScriptsActivate.ps1)
└── ui/
    ├── README.md
    └── streamlit_app.py
```

## 👨‍💻 Built by Anish
