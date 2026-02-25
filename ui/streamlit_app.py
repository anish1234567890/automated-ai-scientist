import streamlit as st
import sys
import os
import json
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────
# Resolve ai_scientist absolute path and insert ONCE at front of sys.path
AI_SCIENTIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai_scientist"))
if AI_SCIENTIST_DIR not in sys.path:
    sys.path.insert(0, AI_SCIENTIST_DIR)
# chdir so that config.py DB_PATH / DATA_PATH resolve relative to ai_scientist/
os.chdir(AI_SCIENTIST_DIR)

from app import run_ai_scientist
from core.lab_notebook import get_all_experiments, clear_all_experiments
from config import DATA_PATH, RESULT_PATH, REPORT_PATH

st.set_page_config(
    page_title="Automated AI Scientist",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 8px; font-weight: 600; }
    .best-model {
        background: #eff6ff; border-left: 4px solid #3b82f6;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;
    }
    div[data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧪 AI Scientist")
    st.caption("Hybrid AutoML System")
    st.divider()
    page = st.radio("Navigate", ["🚀 Run Experiment", "📓 Lab Notebook"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Tech Stack**")
    st.caption("LLM: Llama 3.3 70B (Groq)")
    st.caption("Tuning: Optuna TPE")
    st.caption("Models: RF, XGB, LR, SVM")
    st.caption("UI: Streamlit")
    st.divider()
    st.caption("Built by Anish")


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — RUN EXPERIMENT
# ══════════════════════════════════════════════════════════════════
if "Run" in page:

    st.title("🧪 Automated AI Scientist")
    st.markdown("*LLM selects models → Optuna tunes hyperparameters → Science happens*")
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📂 Dataset")
        uploaded = st.file_uploader("Upload your CSV (must have a `target` column)", type=["csv"])

        if uploaded:
            data_dir = os.path.join(AI_SCIENTIST_DIR, "data")
            os.makedirs(data_dir, exist_ok=True)
            save_path = os.path.join(data_dir, "sample.csv")
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            df_preview = pd.read_csv(save_path)
            st.success(f"✅ Uploaded: {df_preview.shape[0]} rows x {df_preview.shape[1]} cols")
            with st.expander("Preview dataset"):
                st.dataframe(df_preview.head(10), use_container_width=True)

        st.subheader("🧠 Experiment Instruction")
        user_prompt = st.text_area(
            "Tell the AI what to do",
            placeholder="Examples:\n- Try random forest and xgboost for best accuracy\n- Compare all models on this regression task\n- Use fast models to predict category",
            height=120,
        )

        n_trials = st.slider(
            "Optuna trials per model", min_value=10, max_value=50, value=25, step=5,
            help="More trials = better tuning but slower. 25 is a good balance."
        )

        run_btn = st.button("🚀 Run AI Scientist", type="primary", use_container_width=True)

    with col2:
        st.subheader("📡 Live Status")
        status_box = st.empty()
        status_box.info("Waiting for experiment to start...")

    # ── Run ───────────────────────────────────────────────────────
    if run_btn:
        if not user_prompt.strip():
            st.warning("Please enter an experiment instruction.")
            st.stop()
        if not os.path.exists(DATA_PATH):
            st.warning("Please upload a dataset first.")
            st.stop()

        status_lines = []

        def update_status(stage, detail=""):
            status_lines.append(f"**{stage}** {detail}")
            status_box.markdown("\n\n".join(status_lines))

        with st.spinner("Running experiment..."):
            output = run_ai_scientist(user_prompt, progress_callback=update_status, n_trials=n_trials)

        status_box.success("Experiment complete!")
        st.divider()

        results     = output["results"]
        insight     = output["insight"]
        report_path = output["report_path"]
        sel_models  = output["selected_models"]
        task        = results.get("task", "unknown")
        metric      = "Accuracy" if task == "classification" else "RMSE"
        models_data = [m for m in results.get("models", []) if m.get("score") is not None]
        final_code  = results.get("final_code", "")

        # ── Summary metrics ───────────────────────────────────────
        st.subheader("📊 Results Summary")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Task Type", task.upper())
        with m2:
            st.metric("Models Tried", len(sel_models))
        with m3:
            if models_data:
                best = models_data[0]
                st.metric(f"Best {metric}", f"{best['score']:.4f}", delta=best["name"])
        with m4:
            shape = results.get("dataset_shape", [])
            st.metric("Dataset", f"{shape[0]}x{shape[1]}" if shape else "N/A")

        st.divider()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Leaderboard", "⚙️ Parameters", "🔬 AI Insights", "💻 Final Code", "📄 Report"
        ])

        # ── Tab 1: Leaderboard ────────────────────────────────────
        with tab1:
            st.subheader(f"🏆 Model Leaderboard ({metric})")
            if models_data:
                best = models_data[0]
                st.markdown(f"""
<div class="best-model">
🥇 <strong>Best Model: {best['name']}</strong> &nbsp;|&nbsp;
{metric}: <strong>{best['score']:.4f}</strong>
</div>
""", unsafe_allow_html=True)

                table_rows = []
                for i, m in enumerate(models_data):
                    medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                    table_rows.append({
                        "Rank":   medal,
                        "Model":  m["name"],
                        metric:   round(m["score"], 4),
                        "Trials": m.get("n_trials", 25),
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

                st.subheader(f"📈 {metric} Comparison")
                chart_df = pd.DataFrame({
                    "Model": [m["name"] for m in models_data],
                    metric:  [m["score"] for m in models_data],
                }).set_index("Model")
                st.bar_chart(chart_df)
            else:
                st.warning("No successful model results.")

        # ── Tab 2: Parameters breakdown ───────────────────────────
        with tab2:
            st.subheader("⚙️ Hyperparameter Results — All Models")
            st.caption(f"Optuna ran {results.get('n_trials_per_model', 25)} trials per model (TPE sampler)")

            if models_data:
                for i, m in enumerate(models_data):
                    medal  = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                    score  = m.get("score", 0)
                    params = m.get("best_params", {})

                    with st.expander(f"{medal} {m['name']}  |  {metric}: {score:.4f}", expanded=(i == 0)):
                        col_a, col_b = st.columns([1, 1])

                        with col_a:
                            st.markdown("**Model Info**")
                            st.markdown(f"- **Name:** `{m['name']}`")
                            st.markdown(f"- **Task:** `{task.upper()}`")
                            st.markdown(f"- **{metric}:** `{score:.4f}`")
                            st.markdown(f"- **Trials run:** `{m.get('n_trials', 25)}`")

                        with col_b:
                            st.markdown("**Best Hyperparameters (Optuna)**")
                            if params:
                                param_rows = [
                                    {"Parameter": k, "Best Value": str(v), "Type": type(v).__name__}
                                    for k, v in params.items()
                                ]
                                st.dataframe(
                                    pd.DataFrame(param_rows),
                                    use_container_width=True,
                                    hide_index=True
                                )
                            else:
                                st.info("No tunable hyperparameters (e.g. Linear Regression fits directly)")

            failed = [m for m in results.get("models", []) if m.get("score") is None]
            if failed:
                st.divider()
                st.markdown("**Failed Models**")
                for m in failed:
                    st.error(f"{m['name']}: {m.get('error', 'Unknown error')}")

        # ── Tab 3: AI Insights ────────────────────────────────────
        with tab3:
            st.subheader("🔬 Scientific Analysis by AI Researcher Agent")
            st.info(insight)

        # ── Tab 4: Final Code ─────────────────────────────────────
        with tab4:
            st.subheader("💻 Final ML Code with Best Hyperparameters")
            st.caption("Fully runnable Python. Uses exact best parameters found by Optuna. Copy or download.")

            if final_code:
                st.code(final_code, language="python")

                st.download_button(
                    label="⬇️ Download final_ml_code.py",
                    data=final_code,
                    file_name="final_ml_code.py",
                    mime="text/x-python",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.warning("No code generated. Make sure the experiment completed successfully.")

        # ── Tab 5: Report ─────────────────────────────────────────
        with tab5:
            st.subheader("📄 Download PDF Report")
            if report_path and os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    file_bytes = f.read()
                ext  = os.path.splitext(report_path)[1]
                mime = "application/pdf" if ext == ".pdf" else "text/plain"
                st.download_button(
                    label=f"⬇️ Download Report ({ext.upper()})",
                    data=file_bytes,
                    file_name=f"ai_scientist_report{ext}",
                    mime=mime,
                    use_container_width=True,
                )
            else:
                st.info("Report will appear here after the experiment completes.")

            st.divider()
            st.subheader("📋 Agent Logs")
            logs_text = "\n".join(output.get("logs", []))
            st.text_area("Full logs", logs_text, height=350)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — LAB NOTEBOOK
# ══════════════════════════════════════════════════════════════════
elif "Notebook" in page:

    st.title("📓 Lab Notebook")
    st.markdown("*Every experiment stored permanently in SQLite.*")
    st.divider()

    experiments = get_all_experiments()

    if not experiments:
        st.info("No experiments recorded yet. Run your first experiment!")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Experiments", len(experiments))
        with c2:
            st.metric("Classification", sum(1 for e in experiments if e.get("task") == "classification"))
        with c3:
            st.metric("Regression", sum(1 for e in experiments if e.get("task") == "regression"))

        st.divider()

        for exp in experiments:
            task      = exp.get("task", "unknown")
            icon      = "🔵" if task == "classification" else "🟠"
            metric    = "Accuracy" if task == "classification" else "RMSE"
            score_str = f"{exp['best_score']:.4f}" if exp.get("best_score") is not None else "N/A"
            label     = f"{exp['user_prompt'][:55]}..." if len(exp['user_prompt']) > 55 else exp['user_prompt']

            with st.expander(f"{icon} [{exp['timestamp']}]  {label}  |  Best: {exp['best_model']} ({score_str})"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Prompt:** {exp['user_prompt']}")
                    st.markdown(f"**Task:** {task.upper()}")
                    st.markdown(f"**Dataset shape:** {exp.get('dataset_shape', 'N/A')}")
                    st.markdown(f"**Models tried:** {', '.join(exp.get('selected_models', []))}")

                with col2:
                    models_data = [m for m in exp["results"].get("models", []) if m.get("score") is not None]
                    if models_data:
                        rows = []
                        for m in models_data:
                            params_str = ", ".join([f"{k}={v}" for k, v in m.get("best_params", {}).items()])
                            rows.append({
                                "Model":       m["name"],
                                metric:        round(m["score"], 4),
                                "Best Params": params_str or "N/A",
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                if exp.get("insight"):
                    st.markdown("**🔬 AI Insight:**")
                    st.info(exp["insight"])

                # Code download from notebook
                final_code = exp["results"].get("final_code", "")
                if final_code:
                    st.download_button(
                        label="⬇️ Download Code from This Experiment",
                        data=final_code,
                        file_name=f"ml_code_exp_{exp['id']}.py",
                        mime="text/x-python",
                        key=f"dl_code_{exp['id']}",
                    )

        st.divider()
        if st.button("🗑️ Clear All Experiments", type="secondary"):
            clear_all_experiments()
            st.rerun()