from core.researcher import decide_models, generate_insight
from core.automl_engine import run_automl
from core.lab_notebook import save_experiment
from core.report_generator import generate_pdf_report
from config import DATA_PATH, RESULT_PATH


def run_ai_scientist(user_prompt: str, progress_callback=None, n_trials: int = 25) -> dict:
    """
    Full pipeline:
      1. LLM decides which models to try
      2. AutoML engine trains + tunes each model with Optuna
      3. LLM generates scientific insight on results
      4. Save to SQLite lab notebook
      5. Generate PDF report
      6. Return structured result dict for the UI
    """

    def log(stage, detail=""):
        if progress_callback:
            progress_callback(stage, detail)

    logs = []

    # Step 1: LLM selects models
    log("🧠 Researcher agent deciding models...")
    selected_models = decide_models(user_prompt)
    logs.append(f"🧠 Models selected by LLM: {', '.join(selected_models)}")

    # Step 2: AutoML engine
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
        return {
            "logs": logs,
            "results": results,
            "insight": "",
            "report_path": "",
            "selected_models": selected_models,
        }

    task = results.get("task", "unknown")
    logs.append(f"✅ Task detected: {task.upper()}")
    logs.append(f"📊 Dataset shape: {results.get('dataset_shape')}")

    for m in results.get("models", []):
        score = m.get("score")
        if score is not None:
            metric = "Accuracy" if task == "classification" else "RMSE"
            logs.append(f"   {m['name']:25s} → {metric}: {score:.4f}  params: {m.get('best_params', {})}")
        else:
            logs.append(f"   {m['name']:25s} → Failed: {m.get('error', 'unknown error')}")

    # Step 3: LLM generates insight
    log("🔬 Generating scientific insights...")
    insight = generate_insight(results, user_prompt)
    logs.append(f"\n🔬 AI Insight:\n{insight}")

    # Step 4: Save to lab notebook
    log("📓 Saving to lab notebook...")
    save_experiment(user_prompt, results, insight, selected_models)
    logs.append("📓 Experiment saved to lab notebook")

    # Step 5: Generate PDF report
    log("📄 Generating PDF report...")
    report_path = generate_pdf_report(results, insight, user_prompt)
    logs.append(f"📄 Report saved: {report_path}")

    log("✅ Complete!")

    return {
        "logs": logs,
        "results": results,
        "insight": insight,
        "report_path": report_path,
        "selected_models": selected_models,
    }


if __name__ == "__main__":
    def printer(stage, detail=""):
        print(stage, detail)
    result = run_ai_scientist("try random forest and xgboost", progress_callback=printer)
    for line in result["logs"]:
        print(line)