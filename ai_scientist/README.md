<!-- v4.0 -->

# ⚙️ `ai_scientist` Backend Reference

This directory contains the orchestration pipeline (`app.py`), all core agents (`core/`), config (`config.py`), dependencies, and sample data.

## 🚀 Install and run standalone

```bash
pip install -r ai_scientist/requirements.txt
python ai_scientist/app.py
```

Set API key in `ai_scientist\.env`:

```env
GROQ_API_KEY=your_groq_key
```

## 🧩 Core modules and public functions

| Module | Public functions |
|---|---|
| `core/automl_engine.py` | `detect_task(y)`, `preprocess(X)`, `run_automl(data_path, selected_models, result_path, n_trials, progress_callback)` |
| `core/researcher.py` | `decide_models(user_prompt, df=None)`, `decide_unsupervised_algos(user_prompt)`, `generate_hypothesis(...)`, `generate_insight(...)`, `generate_unsupervised_insight(...)` |
| `core/feature_engineer.py` | `run_feature_engineering(df, task, user_prompt, max_features=8)` |
| `core/shap_explainer.py` | `compute_shap(model, X_train, X_test, feature_names, model_name, task, max_samples=200)`, `run_shap_for_best_model(results, X, y, task)` |
| `core/self_improve.py` | `build_improved_experiment(plan, original_prompt, selected_models, n_trials, shap_result)`, `compare_rounds(round1, round2)` |
| `core/unsupervised_engine.py` | `should_run_unsupervised(df, user_prompt)`, `preprocess_unsupervised(df)`, `compute_pca_2d(X)`, `compute_pca_variance(X)`, `run_unsupervised(data_path, selected_algos, result_path, n_trials=20, progress_callback=None)` |
| `core/lab_notebook.py` | `init_db()`, `save_experiment(...)`, `get_all_experiments()`, `get_experiment_by_id(exp_id)`, `export_to_csv(output_path=None)`, `print_paper_summary()`, `clear_all_experiments()` |
| `core/report_generator.py` | `generate_pdf_report(results, insight, user_prompt, mode='supervised', ...)` |
| `core/data_health.py` | `run_health_check(df, task=None)` |
| `core/cluster_profiler.py` | `profile_clusters(df_original, labels, feature_names, algo_name='')`, `generate_cluster_narrative(profile)` |
| `core/coder.py` | `generate_code(hypothesis)` |
| `core/__init__.py` | package marker (no functions) |

## 🧭 Pipeline behavior (`app.py`)

```text
run_ai_scientist(...)
  1) Read config.DATA_PATH
  2) run_health_check
  3) Route mode via should_run_unsupervised
  4a) Supervised: decide_models -> hypothesis -> optional feature eng -> run_automl
      -> SHAP -> optional self-improve round2 -> insight -> save_experiment -> report
  4b) Unsupervised: decide_unsupervised_algos -> run_unsupervised
      -> cluster profiling -> insight -> save_experiment -> report
```

## 🛠️ Config values (`config.py`)

| Name | Value / meaning |
|---|---|
| `BASE_DIR` | Absolute path of `ai_scientist` directory |
| `DATA_PATH` | `BASE_DIR\data\sample.csv` |
| `OUTPUT_DIR` | `BASE_DIR\outputs` |
| `RESULT_PATH` | `OUTPUT_DIR\results.json` |
| `DB_PATH` | `OUTPUT_DIR\lab_notebook.db` |
| `REPORT_PATH` | `OUTPUT_DIR\report.pdf` |
| `LOGS_PATH` | `OUTPUT_DIR\logs.txt` |
| `GENERATED_CODE_PATH` | `OUTPUT_DIR\generated_script.py` |
| `MODEL_RESEARCHER` | `llama-3.3-70b-versatile` |
| `MODEL_CODER` | `llama-3.3-70b-versatile` |
| `MAX_RETRIES` | `3` |

## 📦 Output files and where they are saved

| Output | Path | Produced by |
|---|---|---|
| Experiment results JSON | `ai_scientist\outputs\results.json` | `run_automl()` and `run_unsupervised()` |
| SQLite lab notebook | `ai_scientist\outputs\lab_notebook.db` | `save_experiment()` |
| Report PDF | `ai_scientist\outputs\report.pdf` | `generate_pdf_report()` |
| Report fallback text | same path with `.txt` extension | `generate_pdf_report()` when `fpdf2` unavailable |
| Report fallback PDF | `outputs\report_fallback.pdf` | `generate_pdf_report()` secondary fallback |
| Enriched training CSV | `ai_scientist\data\sample_enriched.csv` | `app.py` feature engineering branch |
| Round-2 reduced CSV | `<round1_path>_r2.csv` (example: `sample_enriched_r2.csv`) | `app.py` self-improvement drop-columns branch |
| Paper CSV export | `ai_scientist\outputs\paper_results.csv` (default) | `export_to_csv()` when invoked |

## 🧾 Dataset requirements

| Mode | Requirements |
|---|---|
| Supervised | CSV must contain a `target` column (`run_automl` returns error if missing) |
| Unsupervised | Any CSV; `target` is optional and dropped if present |

Additional format details derived from preprocessing:

1. File type: CSV readable by `pandas.read_csv`.
2. Categorical columns are label-encoded.
3. Missing numeric values are filled with median.
4. Task detection for supervised is based on `target`: object dtype or `<15` unique values → classification, otherwise regression.
