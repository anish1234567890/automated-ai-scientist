import pandas as pd

from core.researcher import (
    decide_models,
    generate_insight,
    decide_unsupervised_algos,
    generate_unsupervised_insight,
)
from core.automl_engine       import run_automl
from core.unsupervised_engine import run_unsupervised, should_run_unsupervised
from core.lab_notebook        import save_experiment
from core.report_generator    import generate_pdf_report
from config import DATA_PATH, RESULT_PATH


def run_ai_scientist(user_prompt: str,
                     progress_callback=None,
                     n_trials: int = 25) -> dict:
    """
    Full pipeline — auto-routes to supervised or unsupervised.

    Supervised   → needs a 'target' column, trains + tunes ML models
    Unsupervised → no 'target' column OR user mentions clustering keywords,
                   runs clustering / anomaly detection algorithms
    """

    def log(stage, detail=""):
        if progress_callback:
            progress_callback(stage, detail)

    logs = []

    # ── Detect mode ───────────────────────────────────────────────
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        return {"logs": [f"❌ Cannot load dataset: {e}"],
                "results": {"error": str(e)}, "insight": "",
                "report_path": "", "selected_models": [], "mode": "unknown"}

    mode = "unsupervised" if should_run_unsupervised(df, user_prompt) else "supervised"
    logs.append(f"🔍 Mode detected: {mode.upper()}")
    log(f"🔍 Mode: {mode.upper()}")

    # ══════════════════════════════════════════════════════════════
    # UNSUPERVISED BRANCH
    # ══════════════════════════════════════════════════════════════
    if mode == "unsupervised":

        log("🧠 Researcher agent selecting algorithms...")
        selected_algos = decide_unsupervised_algos(user_prompt)
        logs.append(f"🧠 Algorithms selected: {', '.join(selected_algos)}")

        log("⚙️ Unsupervised engine starting...")

        def unsu_progress(algo_name, current, total):
            log(f"⚙️ Running [{current}/{total}]: {algo_name}")

        results = run_unsupervised(
            data_path=DATA_PATH,
            selected_algos=selected_algos,
            result_path=RESULT_PATH,
            n_trials=n_trials,
            progress_callback=unsu_progress,
        )

        if "error" in results:
            logs.append(f"❌ Error: {results['error']}")
            return {"logs": logs, "results": results, "insight": "",
                    "report_path": "", "selected_models": selected_algos,
                    "mode": mode}

        for c in results.get("clustering", []):
            sil = c.get("silhouette")
            if sil is not None:
                logs.append(f"   {c['name']:25s} → Silhouette: {sil:.4f}  "
                             f"Clusters: {c.get('n_clusters_found')}  "
                             f"Params: {c.get('best_params', {})}")
            else:
                logs.append(f"   {c['name']:25s} → Failed: {c.get('error')}")

        log("🔬 Generating scientific insights...")
        insight = generate_unsupervised_insight(results, user_prompt)
        logs.append(f"\n🔬 AI Insight:\n{insight}")

        log("📓 Saving to lab notebook...")
        save_experiment(user_prompt, results, insight, selected_algos,
                        mode="unsupervised")
        logs.append("📓 Experiment saved")

        log("📄 Generating PDF report...")
        report_path = generate_pdf_report(results, insight, user_prompt,
                                          mode="unsupervised")
        logs.append(f"📄 Report: {report_path}")

        log("✅ Complete!")
        return {
            "logs":            logs,
            "results":         results,
            "insight":         insight,
            "report_path":     report_path,
            "selected_models": selected_algos,
            "mode":            mode,
        }

    # ══════════════════════════════════════════════════════════════
    # SUPERVISED BRANCH (original)
    # ══════════════════════════════════════════════════════════════
    log("🧠 Researcher agent deciding models...")
    selected_models = decide_models(user_prompt, df)
    logs.append(f"🧠 Models selected: {', '.join(selected_models)}")

    log("⚙️ AutoML engine starting...")

    def automl_progress(model_name, current, total):
        log(f"⚙️ Tuning [{current}/{total}]: {model_name}")

    results = run_automl(
        data_path=DATA_PATH,
        selected_models=selected_models,
        result_path=RESULT_PATH,
        n_trials=n_trials,
        progress_callback=automl_progress,
    )

    if "error" in results:
        logs.append(f"❌ AutoML Error: {results['error']}")
        return {"logs": logs, "results": results, "insight": "",
                "report_path": "", "selected_models": selected_models,
                "mode": mode}

    task = results.get("task", "unknown")
    logs.append(f"✅ Task detected: {task.upper()}")
    logs.append(f"📊 Dataset shape: {results.get('dataset_shape')}")

    metric = "Accuracy" if task == "classification" else "RMSE"
    for m in results.get("models", []):
        score = m.get("score")
        if score is not None:
            logs.append(f"   {m['name']:25s} → {metric}: {score:.4f}  "
                        f"params: {m.get('best_params', {})}")
        else:
            logs.append(f"   {m['name']:25s} → Failed: {m.get('error', '?')}")

    log("🔬 Generating scientific insights...")
    insight = generate_insight(results, user_prompt)
    logs.append(f"\n🔬 AI Insight:\n{insight}")

    log("📓 Saving to lab notebook...")
    save_experiment(user_prompt, results, insight, selected_models,
                    mode="supervised")
    logs.append("📓 Experiment saved")

    log("📄 Generating PDF report...")
    report_path = generate_pdf_report(results, insight, user_prompt,
                                      mode="supervised")
    logs.append(f"📄 Report: {report_path}")

    log("✅ Complete!")
    return {
        "logs":            logs,
        "results":         results,
        "insight":         insight,
        "report_path":     report_path,
        "selected_models": selected_models,
        "mode":            mode,
    }


if __name__ == "__main__":
    def printer(stage, detail=""):
        print(stage, detail)
    result = run_ai_scientist("find clusters in this data", progress_callback=printer)
    for line in result["logs"]:
        print(line)