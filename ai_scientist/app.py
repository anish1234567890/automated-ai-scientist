"""
Automated AI Scientist v3.0  —  app.py
Pipeline orchestrator.

Returns a single dict with ALL keys the UI expects:
  logs, results, results_r2, round_comparison, improvement_plan,
  round2_config, insight, hypothesis, health, shap_result,
  feature_eng_result, report_path, selected_models, mode
"""

import os
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from core.researcher         import (decide_models, generate_insight,
                                      decide_unsupervised_algos,
                                      generate_unsupervised_insight)
from core.automl_engine      import run_automl, preprocess
from core.unsupervised_engine import run_unsupervised, should_run_unsupervised
from core.lab_notebook       import save_experiment
from core.report_generator   import generate_pdf_report
from core.data_health        import run_health_check
from core.shap_explainer     import run_shap_for_best_model
from core.cluster_profiler   import profile_clusters, generate_cluster_narrative
from core.feature_engineer   import run_feature_engineering
from core.self_improve       import (_call_llm_for_improvement_plan,
                                      build_improved_experiment, compare_rounds)
from config import DATA_PATH, RESULT_PATH


# ── helper: build empty return on error ──────────────────────────
def _empty(msg="", health=None, hypothesis="", mode="unknown"):
    return dict(
        logs=[f"❌ {msg}"] if msg else [],
        results={"error": msg},
        results_r2=None,
        round_comparison={},
        improvement_plan={},
        round2_config={},
        insight="",
        hypothesis=hypothesis,
        health=health or {},
        shap_result={},
        feature_eng_result={"new_features": [], "n_added": 0},
        report_path="",
        selected_models=[],
        mode=mode,
    )


# ═════════════════════════════════════════════════════════════════
def run_ai_scientist(
    user_prompt:         str,
    progress_callback    = None,
    n_trials:            int  = 25,
    enable_feature_eng:  bool = True,
    enable_self_improve: bool = True,
) -> dict:

    def log(msg, _=""):
        if progress_callback:
            progress_callback(msg, _)

    logs = []

    # ── 1. Load CSV ───────────────────────────────────────────────
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        return _empty(f"Cannot load dataset: {e}")

    # ── 2. Health check ───────────────────────────────────────────
    log("🩺 Running dataset health check...")
    mode_guess = "unsupervised" if should_run_unsupervised(df, user_prompt) else "supervised"
    task_guess = None
    if mode_guess == "supervised" and "target" in df.columns:
        task_guess = "classification" if df["target"].nunique() < 15 else "regression"

    health = run_health_check(df, task=task_guess)
    logs.append(f"🩺 Health: Grade {health['grade']} ({health['score']}/100) — {health['summary']}")
    for iss in health.get("issues", []):
        if iss["severity"] == "high":
            logs.append(f"   ⚠️ HIGH: {iss['message']}")

    mode = mode_guess
    logs.append(f"🔍 Mode: {mode.upper()}")
    log(f"🔍 Mode: {mode.upper()}")

    # ── Route ─────────────────────────────────────────────────────
    if mode == "unsupervised":
        return _unsupervised_pipeline(df, user_prompt, health, n_trials, logs, log)

    return _supervised_pipeline(
        df, user_prompt, health, task_guess,
        n_trials, enable_feature_eng, enable_self_improve, logs, log
    )


# ═════════════════════════════════════════════════════════════════
# SUPERVISED
# ═════════════════════════════════════════════════════════════════
def _supervised_pipeline(df, user_prompt, health, task_guess,
                          n_trials, enable_fe, enable_si, logs, log):

    import time as _time
    _start_time = _time.time()

    # ── 3. Model selection ────────────────────────────────────────
    log("🧠 Researcher agent deciding models...")
    selected_models = decide_models(user_prompt, df)
    logs.append(f"🧠 Models: {', '.join(selected_models)}")

    # ── 4. Hypothesis ─────────────────────────────────────────────
    log("💡 Generating pre-experiment hypothesis...")
    hypothesis = ""
    try:
        from core.researcher import generate_hypothesis
        hypothesis = generate_hypothesis(user_prompt, selected_models, df, health)
        logs.append(f"💡 Hypothesis: {hypothesis[:140]}...")
    except Exception as e:
        logs.append(f"   ⚠️ Hypothesis skipped: {e}")

    # ── 5. Feature Engineering ────────────────────────────────────
    df_train = df.copy()
    train_path = DATA_PATH
    fe_result = {"new_features": [], "n_added": 0}

    if enable_fe and len(df) < 100_000:
        log("🔧 LLM Feature Engineering...")
        try:
            fe = run_feature_engineering(df, task_guess or "classification", user_prompt)
            if not fe.get("error") and fe.get("n_added", 0) > 0:
                df_train  = fe["df_enriched"]
                fe_result = fe
                added = [f["name"] for f in fe["new_features"] if f["status"] == "added"]
                logs.append(f"🔧 {fe['n_added']} new features: {', '.join(added[:5])}")
                _base, _ext = os.path.splitext(DATA_PATH)
                enriched = _base + "_enriched" + (_ext or ".csv")
                df_train.to_csv(enriched, index=False)
                train_path = enriched
            else:
                logs.append("🔧 Feature engineering: no valid features generated")
        except Exception as e:
            logs.append(f"   ⚠️ Feature eng failed: {e}")
    else:
        logs.append("🔧 Feature engineering: skipped")

    # ── 6. AutoML Round 1 ─────────────────────────────────────────
    log("⚙️ AutoML Round 1 (Optuna + 3-fold CV, poly+select hoisted)...")

    def _prog(name, i=None, total=None):
        if i is not None and total is not None:
            log(f"⚙️ Tuning [{i}/{total}]: {name}")
        else:
            log(f"   {name}")

    r1 = run_automl(train_path, selected_models, RESULT_PATH, n_trials, _prog)

    if "error" in r1:
        logs.append(f"❌ AutoML Error: {r1['error']}")
        return _empty(r1["error"], health=health, hypothesis=hypothesis, mode="supervised")

    task   = r1.get("task", "unknown")
    metric = "Accuracy" if task == "classification" else "RMSE"
    logs.append(f"✅ Task: {task.upper()} | Shape: {r1.get('dataset_shape')}")

    for m in r1.get("models", []):
        s = m.get("score")
        logs.append(f"   {m['name']:25s} → {f'{metric}: {s:.4f}' if s is not None else 'Failed'}")

    ens = r1.get("ensemble", {})
    if ens and not ens.get("error"):
        logs.append(f"   🤝 Ensemble: {metric}={ens.get('cv_score','?')}")

    # ── 7. SHAP ───────────────────────────────────────────────────
    # SHAP runs ONLY on the best model (the first in the sorted leaderboard).
    # selected_feature_names from run_automl gives the post-poly/select names
    # so SHAP correctly describes which transformed features matter.
    log("🔍 Computing SHAP feature importance...")
    shap_result = {}
    try:
        target_col = "target" if "target" in df_train.columns else df_train.columns[-1]
        X_p = preprocess(df_train.drop(columns=[target_col]))
        y_p = df_train[target_col].copy()
        if y_p.dtype == "object":
            y_p = pd.Series(LabelEncoder().fit_transform(y_p))

        # Only call SHAP once on the single best model (valid[0])
        valid_models = [m for m in r1.get("models", []) if m.get("score") is not None]
        if not valid_models:
            logs.append("   ⚠️ SHAP: No valid models to explain")
        else:
            shap_result = run_shap_for_best_model(r1, X_p, y_p, task)
            if not shap_result.get("error"):
                top3 = shap_result.get("top_features", [])[:3]
                logs.append("🔍 SHAP top: " + ", ".join(
                    f"{t['feature']}({t['importance']:.4f})" for t in top3))
                r1["shap"] = shap_result
            else:
                logs.append(f"   ⚠️ SHAP: {shap_result['error']}")
    except Exception as e:
        logs.append(f"   ⚠️ SHAP failed: {e}")

    # ── 8. Self-Improvement ───────────────────────────────────────
    r2 = None
    improvement_plan = {}
    round_comparison = {}
    round2_config    = {}

    if enable_si:
        log("🔄 Self-improvement agent analyzing Round 1...")
        try:
            improvement_plan = _call_llm_for_improvement_plan(
                r1, health, shap_result, user_prompt, round_num=2
            )
            logs.append(f"🔄 Plan: {improvement_plan.get('reasoning','')[:140]}...")

            round2_config = build_improved_experiment(
                improvement_plan, user_prompt, selected_models, n_trials, shap_result
            )
            for ch in round2_config.get("changes_applied", []):
                logs.append(f"   {ch}")

            train_path_r2 = train_path
            cols_to_drop = round2_config.get("drop_columns", [])
            if cols_to_drop:
                drop_existing = [
                    c for c in cols_to_drop
                    if c in df_train.columns and c != "target"
                ]
                if drop_existing:
                    df_r2 = df_train.drop(columns=drop_existing)
                    _base, _ext = os.path.splitext(train_path)
                    train_path_r2 = _base + "_r2" + (_ext or ".csv")
                    df_r2.to_csv(train_path_r2, index=False)
                    logs.append(f"🎯 Round 2 dropped columns: {', '.join(drop_existing[:8])}")

            log("⚙️ AutoML Round 2 (auto-improved)...")
            r2 = run_automl(train_path_r2, round2_config["selected_models"],
                            RESULT_PATH, round2_config["n_trials"], _prog)

            if "error" not in r2:
                round_comparison = compare_rounds(r1, r2)
                imp  = round_comparison.get("improved", False)
                r2b  = round_comparison.get("round2_best", {})
                logs.append(
                    f"🔄 Round 2: {r2b.get('name','?')} {metric}="
                    f"{r2b.get('score',0):.4f} "
                    f"({'✅ improved' if imp else '⚠️ no improvement'}, "
                    f"Δ={round_comparison.get('delta',0):+.4f})"
                )
                # SHAP for round 2
                try:
                    sr2 = run_shap_for_best_model(r2, X_p, y_p, task)
                    if not sr2.get("error"):
                        r2["shap"] = sr2
                except Exception as e:
                    logs.append(f"⚠️ Round 2 SHAP failed: {e}")
            else:
                logs.append(f"   ⚠️ Round 2 failed: {r2['error']}")
                r2 = None
        except Exception as e:
            logs.append(f"   ⚠️ Self-improvement failed: {e}")

    # ── 9. Insight ────────────────────────────────────────────────
    log("🔬 Generating scientific insights...")
    try:
        insight = generate_insight(
            results    =r1,
            user_prompt=user_prompt,
            hypothesis =hypothesis,
            shap_result=shap_result if not shap_result.get("error") else None,
        )
    except Exception:
        # fallback if generate_insight doesn't accept new kwargs
        insight = generate_insight(r1, user_prompt)
    logs.append(f"\n🔬 Insight: {insight[:120]}...")

    # ── 10. Save + Report ─────────────────────────────────────────
    _elapsed = _time.time() - _start_time
    _dataset_name = os.path.splitext(os.path.basename(DATA_PATH))[0]
    _health_grade = health.get("grade")
    _features_added = fe_result.get("n_added", 0)
    _shap_expl = (shap_result or {}).get("explainer_type")
    _ensemble = r1.get("ensemble") or {}
    _ens_score = _ensemble.get("cv_score")
    if _ens_score is None:
        _ens_score = _ensemble.get("score")
    if _ens_score is None:
        _ens_score = r1.get("ensemble_score")
    _r1b = max(
        [m for m in r1.get("models", []) if m.get("score") is not None],
        key=lambda x: x["score"], default={}
    ) if r1.get("task") == "classification" else min(
        [m for m in r1.get("models", []) if m.get("score") is not None],
        key=lambda x: x["score"], default={}
    )
    _r1_model = _r1b.get("name")
    _r1_score = _r1b.get("score")
    _r2b = round_comparison.get("round2_best", {}) if r2 else {}
    _r2_model = _r2b.get("name") if r2 else None
    _r2_score = _r2b.get("score") if r2 else None
    _delta = round_comparison.get("delta") if r2 else None
    log("📓 Saving to lab notebook...")
    save_experiment(
        user_prompt, r1, insight, selected_models, mode="supervised",
        round1_score=_r1_score, round1_model=_r1_model,
        round2_score=_r2_score, round2_model=_r2_model,
        delta=_delta, shap_explainer=_shap_expl,
        ensemble_score=_ens_score, health_grade=_health_grade,
        experiment_time_sec=_elapsed, dataset_name=_dataset_name,
        features_added=_features_added,
    )
    if r2 and "error" not in r2:
        save_experiment(
            user_prompt + " [v3 Round 2]", r2,
            f"[Auto-improved] {insight[:200]}",
            round2_config.get("selected_models", selected_models),
            mode="supervised",
        )
    logs.append("📓 Saved")

    log("📄 Generating PDF report...")
    try:
        report_path = generate_pdf_report(
            r1, insight, user_prompt, mode="supervised",
            dataset_name=_dataset_name,
            health_grade=_health_grade,
            round1_score=_r1_score,
            round1_model=_r1_model,
            round2_score=_r2_score,
            round2_model=_r2_model,
            delta=_delta,
            shap_explainer=_shap_expl,
            ensemble_score=_ens_score,
            experiment_time_sec=_elapsed,
            features_added=_features_added,
        )
    except Exception as e:
        logs.append(f"   ⚠️ PDF failed: {e}")
        report_path = ""

    logs.append(f"📄 Report: {report_path}")
    log("✅ Complete!")

    return dict(
        logs             = logs,
        results          = r1,
        results_r2       = r2,
        round_comparison = round_comparison,
        improvement_plan = improvement_plan,
        round2_config    = round2_config,
        insight          = insight,
        hypothesis       = hypothesis,
        health           = health,
        shap_result      = shap_result,
        feature_eng_result = fe_result,
        report_path      = report_path,
        selected_models  = selected_models,
        mode             = "supervised",
    )


# ═════════════════════════════════════════════════════════════════
# UNSUPERVISED
# ═════════════════════════════════════════════════════════════════
def _unsupervised_pipeline(df, user_prompt, health, n_trials, logs, log):

    log("🧠 Researcher agent selecting algorithms...")
    selected_algos = decide_unsupervised_algos(user_prompt)
    logs.append(f"🧠 Algorithms: {', '.join(selected_algos)}")

    log("⚙️ Unsupervised engine starting...")

    def _uprog(name, i, total):
        log(f"⚙️ Running [{i}/{total}]: {name}")

    results = run_unsupervised(DATA_PATH, selected_algos, RESULT_PATH, n_trials, _uprog)

    if "error" in results:
        logs.append(f"❌ Error: {results['error']}")
        return _empty(results["error"], health=health, mode="unsupervised")

    for c in results.get("clustering", []):
        sil = c.get("silhouette")
        logs.append(
            f"   {c['name']:25s} → "
            f"{'Silhouette: ' + f'{sil:.4f}' if sil is not None else 'Failed'}"
            f"  Clusters: {c.get('n_clusters_found','?')}"
        )

    # Cluster profiling
    cluster_profile   = {}
    cluster_narrative = ""
    best_labels       = results.get("best_labels", [])
    if best_labels and len(best_labels) == len(df):
        try:
            cluster_profile   = profile_clusters(
                df, best_labels,
                results.get("feature_names", []),
                results.get("best_algo", ""),
            )
            cluster_narrative = generate_cluster_narrative(cluster_profile)
            results["cluster_profile"] = cluster_profile
            logs.append(f"🔍 {cluster_profile.get('n_clusters')} cluster profiles built")
        except Exception as e:
            logs.append(f"   ⚠️ Cluster profiling failed: {e}")

    log("🔬 Generating scientific insights...")
    try:
        insight = generate_unsupervised_insight(results, user_prompt, cluster_narrative)
    except Exception:
        insight = generate_unsupervised_insight(results, user_prompt)
    logs.append(f"\n🔬 Insight: {insight[:120]}...")

    log("📓 Saving to lab notebook...")
    save_experiment(user_prompt, results, insight, selected_algos, mode="unsupervised")
    logs.append("📓 Saved")

    log("📄 Generating PDF report...")
    try:
        report_path = generate_pdf_report(results, insight, user_prompt, mode="unsupervised")
    except Exception as e:
        logs.append(f"   ⚠️ PDF failed: {e}")
        report_path = ""
    logs.append(f"📄 Report: {report_path}")
    log("✅ Complete!")

    return dict(
        logs             = logs,
        results          = results,
        results_r2       = None,
        round_comparison = {},
        improvement_plan = {},
        round2_config    = {},
        insight          = insight,
        hypothesis       = "",
        health           = health,
        shap_result      = {},
        feature_eng_result = {"new_features": [], "n_added": 0},
        report_path      = report_path,
        selected_models  = selected_algos,
        mode             = "unsupervised",
    )


if __name__ == "__main__":
    def _p(s, d=""): print(s, d)
    out = run_ai_scientist("try random forest and xgboost", _p)
    for l in out["logs"]: print(l)
