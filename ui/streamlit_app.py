import streamlit as st
import sys
import os
import json
import pandas as pd
import numpy as np

# ── Path setup ────────────────────────────────────────────────────
AI_SCIENTIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai_scientist"))
if AI_SCIENTIST_DIR not in sys.path:
    sys.path.insert(0, AI_SCIENTIST_DIR)
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
    .unsup-card {
        background: #f0fdf4; border-left: 4px solid #22c55e;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;
    }
    .anomaly-card {
        background: #fef2f2; border-left: 4px solid #ef4444;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;
    }
    div[data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧪 AI Scientist")
    st.caption("Hybrid AutoML + Unsupervised System")
    st.divider()
    page = st.radio("Navigate", ["🚀 Run Experiment", "📓 Lab Notebook"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Supervised (21 models)**")
    st.caption("RF, XGB, LightGBM, Ridge, Lasso, SVM, KNN...")
    st.markdown("**Unsupervised (10 algorithms)**")
    st.caption("K-Means, DBSCAN, GMM, Iso Forest, PCA...")
    st.divider()
    st.markdown("**Tech Stack**")
    st.caption("LLM: Llama 3.3 70B (Groq)")
    st.caption("Tuning: Optuna TPE")
    st.caption("UI: Streamlit")
    st.divider()
    st.caption("Built by Anish")


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — RUN EXPERIMENT
# ══════════════════════════════════════════════════════════════════
if "Run" in page:

    st.title("🧪 Automated AI Scientist")
    st.markdown("*LLM selects models → Optuna tunes → Science happens. Supports supervised & unsupervised.*")
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📂 Dataset")
        st.caption("For **supervised**: CSV must have a `target` column. For **unsupervised**: any CSV works.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded:
            data_dir = os.path.join(AI_SCIENTIST_DIR, "data")
            os.makedirs(data_dir, exist_ok=True)
            save_path = os.path.join(data_dir, "sample.csv")
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            df_preview = pd.read_csv(save_path)
            has_target = "target" in df_preview.columns
            if has_target:
                st.success(f"✅ {df_preview.shape[0]} rows × {df_preview.shape[1]} cols — **target column found** (supervised ready)")
            else:
                st.info(f"ℹ️ {df_preview.shape[0]} rows × {df_preview.shape[1]} cols — **no target column** (will run unsupervised)")
            with st.expander("Preview dataset"):
                st.dataframe(df_preview.head(10), use_container_width=True)

        st.subheader("🧠 Experiment Instruction")
        user_prompt = st.text_area(
            "Tell the AI what to do",
            placeholder=(
                "Supervised examples:\n"
                "  • Try random forest and xgboost\n"
                "  • Compare ridge and lasso regression\n\n"
                "Unsupervised examples:\n"
                "  • Find clusters in this data\n"
                "  • Detect anomalies and outliers\n"
                "  • Segment customers into groups\n"
                "  • Try kmeans and dbscan"
            ),
            height=140,
        )

        n_trials = st.slider(
            "Optuna trials per algorithm", min_value=10, max_value=50, value=25, step=5,
            help="More trials = better tuning but slower."
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

        status_box.success("✅ Experiment complete!")
        st.divider()

        results     = output["results"]
        insight     = output["insight"]
        report_path = output["report_path"]
        sel_models  = output["selected_models"]
        mode        = output.get("mode", "supervised")
        shape       = results.get("dataset_shape", [])
        final_code  = results.get("final_code", "")

        # ══════════════════════════════════════════════════════════
        # SUPERVISED RESULTS
        # ══════════════════════════════════════════════════════════
        if mode == "supervised":
            task        = results.get("task", "unknown")
            metric      = "Accuracy" if task == "classification" else "RMSE"
            models_data = [m for m in results.get("models", []) if m.get("score") is not None]

            st.subheader("📊 Results Summary")
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Mode", "SUPERVISED")
            with m2: st.metric("Task", task.upper())
            with m3:
                if models_data:
                    best = models_data[0]
                    st.metric(f"Best {metric}", f"{best['score']:.4f}", delta=best["name"])
            with m4: st.metric("Dataset", f"{shape[0]}×{shape[1]}" if shape else "N/A")

            st.divider()
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🏆 Leaderboard", "⚙️ Parameters", "🔬 AI Insights", "💻 Final Code", "📄 Report"
            ])

            with tab1:
                st.subheader(f"🏆 Model Leaderboard ({metric})")
                if models_data:
                    best = models_data[0]
                    st.markdown(f'<div class="best-model">🥇 <strong>Best: {best["name"]}</strong> &nbsp;|&nbsp; {metric}: <strong>{best["score"]:.4f}</strong></div>', unsafe_allow_html=True)
                    rows = []
                    for i, m in enumerate(models_data):
                        medal = ["🥇","🥈","🥉"][i] if i < 3 else f"#{i+1}"
                        rows.append({"Rank": medal, "Model": m["name"], metric: round(m["score"],4), "Trials": m.get("n_trials",25)})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    chart_df = pd.DataFrame({"Model":[m["name"] for m in models_data], metric:[m["score"] for m in models_data]}).set_index("Model")
                    st.bar_chart(chart_df)
                else:
                    st.warning("No successful results.")

            with tab2:
                st.subheader("⚙️ Hyperparameter Results")
                st.caption(f"Optuna: {results.get('n_trials_per_model',25)} trials per model, TPE sampler")
                for i, m in enumerate(models_data):
                    medal = ["🥇","🥈","🥉"][i] if i < 3 else f"#{i+1}"
                    with st.expander(f"{medal} {m['name']}  |  {metric}: {m.get('score',0):.4f}", expanded=(i==0)):
                        ca, cb = st.columns(2)
                        with ca:
                            st.markdown(f"**Name:** `{m['name']}`  \n**Task:** `{task.upper()}`  \n**{metric}:** `{m.get('score',0):.4f}`  \n**Trials:** `{m.get('n_trials',25)}`")
                        with cb:
                            params = m.get("best_params", {})
                            if params:
                                st.dataframe(pd.DataFrame([{"Parameter":k,"Best Value":str(v),"Type":type(v).__name__} for k,v in params.items()]), use_container_width=True, hide_index=True)
                            else:
                                st.info("No tunable hyperparameters")
                failed = [m for m in results.get("models",[]) if m.get("score") is None]
                if failed:
                    st.divider()
                    for m in failed:
                        st.error(f"{m['name']}: {m.get('error','Unknown error')}")

            with tab3:
                st.subheader("🔬 Scientific Analysis")
                st.info(insight)

            with tab4:
                st.subheader("💻 Final Code with Best Hyperparameters")
                if final_code:
                    st.code(final_code, language="python")
                    st.download_button("⬇️ Download final_ml_code.py", data=final_code,
                        file_name="final_ml_code.py", mime="text/x-python",
                        use_container_width=True, type="primary")
                else:
                    st.warning("No code generated.")

            with tab5:
                st.subheader("📄 Download PDF Report")
                if report_path and os.path.exists(report_path):
                    with open(report_path, "rb") as f: file_bytes = f.read()
                    ext = os.path.splitext(report_path)[1]
                    st.download_button(f"⬇️ Download Report ({ext.upper()})", data=file_bytes,
                        file_name=f"report{ext}", mime="application/pdf" if ext==".pdf" else "text/plain",
                        use_container_width=True)
                else:
                    st.info("Report not generated yet.")
                st.divider()
                st.subheader("📋 Agent Logs")
                st.text_area("Logs", "\n".join(output.get("logs",[])), height=300)

        # ══════════════════════════════════════════════════════════
        # UNSUPERVISED RESULTS
        # ══════════════════════════════════════════════════════════
        else:
            # correct field: silhouette (not score)
            all_cls     = results.get("clustering", [])
            clustering  = [c for c in all_cls if c.get("silhouette") is not None]
            # Isolation Forest & LOF live in clustering list too — split by name
            anomaly_names = {"isolation forest", "local outlier factor", "lof", "anomaly detection"}
            anomalies   = [c for c in clustering if c["name"].lower() in anomaly_names]
            clustering  = [c for c in clustering if c["name"].lower() not in anomaly_names]
            pca_coords  = results.get("pca_coords", [])
            best_labels = results.get("best_labels", [])

            all_valid    = [c for c in results.get("clustering",[]) if c.get("silhouette") is not None]
            best_overall = max(all_valid, key=lambda x: x["silhouette"]) if all_valid else None

            st.subheader("📊 Results Summary")
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Mode", "UNSUPERVISED")
            with m2: st.metric("Algorithms Run", len(sel_models))
            with m3:
                if best_overall:
                    st.metric("Best Silhouette", f"{best_overall['silhouette']:.4f}", delta=best_overall["name"])
            with m4: st.metric("Dataset", f"{shape[0]}×{shape[1]}" if shape else "N/A")

            st.divider()
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🔵 Clusters", "🔴 Anomalies", "🔬 AI Insights", "💻 Final Code", "📄 Report"
            ])

            # ── Tab 1: Clustering results + PCA scatter ───────────
            with tab1:
                st.subheader("🔵 Clustering Leaderboard")
                st.caption("Sorted by Silhouette Score (higher = better separated clusters)")

                if clustering:
                    best_c = clustering[0]
                    st.markdown(
                        f'<div class="unsup-card">🥇 <strong>Best: {best_c["name"]}</strong> &nbsp;|&nbsp; '
                        f'Silhouette: <strong>{best_c["silhouette"]:.4f}</strong> &nbsp;|&nbsp; '
                        f'Clusters: <strong>{best_c.get("n_clusters_found","?")}</strong></div>',
                        unsafe_allow_html=True
                    )

                    rows = []
                    for i, c in enumerate(clustering):
                        medal = ["🥇","🥈","🥉"][i] if i < 3 else f"#{i+1}"
                        rows.append({
                            "Rank": medal, "Algorithm": c["name"],
                            "Silhouette ↑": round(c.get("silhouette") or 0, 4),
                            "Davies-Bouldin ↓": round(c.get("davies_bouldin") or 0, 4),
                            "Calinski-Harabasz ↑": round(c.get("calinski_harabasz") or 0, 1),
                            "Clusters Found": c.get("n_clusters_found", "?"),
                            "Noise Pts": c.get("n_noise_points", 0),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    # ── PCA 2D Scatter plot ───────────────────────
                    if pca_coords and best_labels:
                        st.subheader("🗺️ PCA Cluster Visualization")
                        st.caption(f"2D PCA projection — colored by {best_c['name']} cluster assignments")
                        pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"])
                        pca_df["Cluster"] = [str(l) for l in best_labels]
                        # Show using altair for colored scatter
                        try:
                            import altair as alt
                            chart = alt.Chart(pca_df).mark_circle(size=60, opacity=0.7).encode(
                                x=alt.X("PC1:Q", title="Principal Component 1"),
                                y=alt.Y("PC2:Q", title="Principal Component 2"),
                                color=alt.Color("Cluster:N", title="Cluster"),
                                tooltip=["PC1", "PC2", "Cluster"]
                            ).properties(
                                width="container", height=400,
                                title=f"Clusters found by {best_c['name']}"
                            ).interactive()
                            st.altair_chart(chart, use_container_width=True)
                        except ImportError:
                            # Fallback: plain scatter
                            st.scatter_chart(pca_df, x="PC1", y="PC2", color="Cluster")

                        pca_var = results.get("pca_variance", [])
                        if pca_var:
                            total_var = round(sum(pca_var)*100, 1)
                            st.caption(f"PCA explains {total_var}% of total variance  |  PC1: {round(pca_var[0]*100,1)}%  PC2: {round(pca_var[1]*100,1) if len(pca_var)>1 else 'N/A'}%")

                    # ── Per-algorithm parameter breakdown ─────────
                    st.subheader("⚙️ Hyperparameters per Algorithm")
                    for i, c in enumerate(clustering):
                        medal = ["🥇","🥈","🥉"][i] if i < 3 else f"#{i+1}"
                        with st.expander(f"{medal} {c['name']}  |  Silhouette: {c.get('silhouette',0):.4f}  |  Clusters: {c.get('n_clusters_found','?')}", expanded=(i==0)):
                            ca, cb = st.columns(2)
                            with ca:
                                st.markdown(f"**Algorithm:** `{c['name']}`")
                                st.markdown(f"**Silhouette:** `{c.get('silhouette','N/A')}` *(higher = better, max 1.0)*")
                                st.markdown(f"**Davies-Bouldin:** `{c.get('davies_bouldin','N/A')}` *(lower = better)*")
                                st.markdown(f"**Calinski-Harabasz:** `{c.get('calinski_harabasz','N/A')}` *(higher = better)*")
                                st.markdown(f"**Clusters found:** `{c.get('n_clusters_found','?')}`")
                                st.markdown(f"**Noise points:** `{c.get('n_noise_points',0)}`")
                            with cb:
                                params = c.get("best_params", {})
                                if params:
                                    st.markdown("**Best Hyperparameters (Optuna):**")
                                    st.dataframe(pd.DataFrame([{"Parameter":k,"Best Value":str(v)} for k,v in params.items()]), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No tunable hyperparameters")

                failed_cls = [c for c in results.get("clustering",[]) if c.get("silhouette") is None]
                if failed_cls:
                    st.divider()
                    for c in failed_cls:
                        st.error(f"{c['name']}: {c.get('error','Unknown error')}")

            # ── Tab 2: Anomaly Detection ──────────────────────────
            with tab2:
                st.subheader("🔴 Anomaly Detection Results")
                st.caption("Isolation Forest and Local Outlier Factor label each point as Normal (0) or Anomaly (1)")
                if anomalies:
                    for a in anomalies:
                        params   = a.get("best_params", {})
                        n_anom   = params.get("anomalies_found") or params.get("outliers_found", "?")
                        n_total  = shape[0] if shape else "?"
                        sil      = a.get("silhouette")
                        sil_str  = f"{sil:.4f}" if sil is not None else "N/A"
                        st.markdown(
                            f'<div style="background:#fff0f0;border-left:4px solid #ef4444;padding:10px 14px;'
                            f'border-radius:6px;margin-bottom:10px">'
                            f'🔴 <strong>{a["name"]}</strong> &nbsp;|&nbsp; '
                            f'Anomalies detected: <strong>{n_anom}</strong> / {n_total} &nbsp;|&nbsp; '
                            f'Silhouette: <strong>{sil_str}</strong></div>',
                            unsafe_allow_html=True
                        )
                        display_params = {k:v for k,v in params.items()
                                          if k not in ("anomalies_found","outliers_found")}
                        if display_params:
                            st.markdown("**Best Optuna hyperparameters:**")
                            st.dataframe(
                                pd.DataFrame([{"Parameter":k,"Best Value":str(v)}
                                              for k,v in display_params.items()]),
                                use_container_width=True, hide_index=True
                            )
                        # PCA scatter for anomalies
                        if pca_coords:
                            # labels: 1=anomaly, 0=normal (as set in engine)
                            lbl_key = best_labels  # use best algo labels as fallback
                            st.caption(f"PCA scatter — anomalies vs normal points ({a['name']})")
                            adf = pd.DataFrame(pca_coords, columns=["PC1","PC2"])
                            adf["Type"] = ["Anomaly" if l==1 else "Normal" for l in lbl_key]
                            try:
                                import altair as alt
                                chart = alt.Chart(adf).mark_circle(size=50, opacity=0.7).encode(
                                    x="PC1:Q", y="PC2:Q",
                                    color=alt.Color("Type:N", scale=alt.Scale(
                                        domain=["Normal","Anomaly"], range=["#3b82f6","#ef4444"])),
                                    tooltip=["PC1","PC2","Type"]
                                ).properties(width="container", height=350).interactive()
                                st.altair_chart(chart, use_container_width=True)
                            except ImportError:
                                adf["ColorCode"] = [1 if t=="Anomaly" else 0 for t in adf["Type"]]
                                st.scatter_chart(adf, x="PC1", y="PC2", color="ColorCode")
                        st.divider()
                else:
                    st.info("No anomaly detection algorithms were run. Add **'Isolation Forest'** or **'Local Outlier Factor'** to your prompt — e.g. *'find anomalies using isolation forest'*")

            # ── Tab 3: AI Insights ────────────────────────────────
            with tab3:
                st.subheader("🔬 Scientific Analysis by AI Researcher Agent")
                st.info(insight)

            # ── Tab 4: Final Code ─────────────────────────────────
            with tab4:
                st.subheader("💻 Final Unsupervised Code")
                st.caption("Fully runnable Python with Optuna-tuned parameters + PCA visualization.")
                if final_code:
                    st.code(final_code, language="python")
                    st.download_button("⬇️ Download final_unsupervised_code.py", data=final_code,
                        file_name="final_unsupervised_code.py", mime="text/x-python",
                        use_container_width=True, type="primary")
                else:
                    st.warning("No code generated.")

            # ── Tab 5: Report ─────────────────────────────────────
            with tab5:
                st.subheader("📄 Download PDF Report")
                if report_path and os.path.exists(report_path):
                    with open(report_path, "rb") as f: file_bytes = f.read()
                    ext = os.path.splitext(report_path)[1]
                    st.download_button(f"⬇️ Download Report ({ext.upper()})", data=file_bytes,
                        file_name=f"report{ext}", mime="application/pdf" if ext==".pdf" else "text/plain",
                        use_container_width=True)
                else:
                    st.info("Report not generated yet.")
                st.divider()
                st.subheader("📋 Agent Logs")
                st.text_area("Logs", "\n".join(output.get("logs",[])), height=300)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — LAB NOTEBOOK
# ══════════════════════════════════════════════════════════════════
elif "Notebook" in page:

    st.title("📓 Lab Notebook")
    st.markdown("*Every experiment stored permanently in SQLite — both supervised and unsupervised.*")
    st.divider()

    experiments = get_all_experiments()

    if not experiments:
        st.info("No experiments recorded yet. Run your first experiment!")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total", len(experiments))
        with c2: st.metric("Supervised", sum(1 for e in experiments if e.get("mode","supervised")=="supervised"))
        with c3: st.metric("Unsupervised", sum(1 for e in experiments if e.get("mode")=="unsupervised"))
        with c4:
            cls_c = sum(1 for e in experiments if e.get("task")=="classification")
            st.metric("Classification", cls_c)

        st.divider()

        for exp in experiments:
            mode      = exp.get("mode", "supervised")
            task      = exp.get("task", "unknown")
            score_str = f"{exp['best_score']:.4f}" if exp.get("best_score") is not None else "N/A"
            label     = exp['user_prompt'][:60] + ("..." if len(exp['user_prompt'])>60 else "")

            if mode == "unsupervised":
                icon = "🟢"
                badge = f"UNSUPERVISED | Best: {exp['best_model']} (Silhouette: {score_str})"
            elif task == "classification":
                icon = "🔵"
                badge = f"CLASSIFICATION | Best: {exp['best_model']} (Acc: {score_str})"
            else:
                icon = "🟠"
                badge = f"REGRESSION | Best: {exp['best_model']} (RMSE: {score_str})"

            with st.expander(f"{icon} [{exp['timestamp']}]  {label}  →  {badge}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Prompt:** {exp['user_prompt']}")
                    st.markdown(f"**Mode:** `{mode.upper()}`")
                    st.markdown(f"**Task:** `{task.upper()}`")
                    st.markdown(f"**Dataset shape:** {exp.get('dataset_shape','N/A')}")
                    st.markdown(f"**Algorithms:** {', '.join(exp.get('selected_models',[]))}")

                with col2:
                    exp_results = exp.get("results", {})
                    if mode == "unsupervised":
                        cls_data = [c for c in exp_results.get("clustering",[]) if c.get("silhouette") is not None]
                        if cls_data:
                            rows = []
                            for c in cls_data:
                                params  = c.get("best_params", {})
                                n_anom  = params.get("anomalies_found") or params.get("outliers_found")
                                row = {
                                    "Algorithm":     c["name"],
                                    "Silhouette ↑":  round(c.get("silhouette") or 0, 4),
                                    "DB Index ↓":    round(c.get("davies_bouldin") or 0, 4),
                                    "Clusters Found": c.get("n_clusters_found", "?"),
                                    "Noise Points":   c.get("n_noise_points", 0),
                                }
                                if n_anom is not None:
                                    row["Anomalies"] = n_anom
                                rows.append(row)
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        metric = "Accuracy" if task=="classification" else "RMSE"
                        models_data = [m for m in exp_results.get("models",[]) if m.get("score") is not None]
                        if models_data:
                            rows = [{"Model": m["name"], metric: round(m["score"],4)} for m in models_data]
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                if exp.get("insight"):
                    st.markdown("**🔬 AI Insight:**")
                    st.info(exp["insight"])

                final_code = exp_results.get("final_code", "")
                if final_code:
                    fname = f"{'unsupervised' if mode=='unsupervised' else 'ml'}_code_exp_{exp['id']}.py"
                    st.download_button(
                        label="⬇️ Download Code",
                        data=final_code,
                        file_name=fname,
                        mime="text/x-python",
                        key=f"dl_{exp['id']}",
                    )

        st.divider()
        if st.button("🗑️ Clear All Experiments", type="secondary"):
            clear_all_experiments()
            st.rerun()