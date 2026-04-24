import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

# ── Path setup ────────────────────────────────────────────────────
AI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai_scientist"))
if AI_DIR not in sys.path:
    sys.path.insert(0, AI_DIR)
os.chdir(AI_DIR)

from app import run_ai_scientist
from core.lab_notebook import get_all_experiments, clear_all_experiments
from config import DATA_PATH, REPORT_PATH

st.set_page_config(
    page_title="Automated AI Scientist v3.0",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stButton>button{width:100%;border-radius:8px;font-weight:600;}
.c-blue  {background:#eff6ff;border-left:4px solid #3b82f6;padding:12px 16px;border-radius:6px;margin-bottom:10px;}
.c-yellow{background:#fefce8;border-left:4px solid #eab308;padding:12px 16px;border-radius:6px;margin-bottom:10px;}
.c-green {background:#f0fdf4;border-left:4px solid #22c55e;padding:12px 16px;border-radius:6px;margin-bottom:10px;}
.c-red   {background:#fef2f2;border-left:4px solid #ef4444;padding:12px 16px;border-radius:6px;margin-bottom:10px;}
.c-purple{background:#f5f3ff;border-left:4px solid #8b5cf6;padding:12px 16px;border-radius:6px;margin-bottom:10px;}
div[data-testid="stExpander"]{border:1px solid #e2e8f0;border-radius:8px;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧪 AI Scientist v3.0")
    st.caption("Autonomous ML Research System")
    st.divider()
    page = st.radio("Navigate",
                    ["🚀 Run Experiment", "📓 Lab Notebook"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("**v3.0 Features**")
    st.caption("🩺 Dataset Health Check")
    st.caption("💡 Pre-Experiment Hypothesis")
    st.caption("🔧 LLM Feature Engineering")
    st.caption("⚙️ AutoML — 21 models, 5-fold CV")
    st.caption("🔍 SHAP Feature Importance")
    st.caption("🔄 Iterative Self-Improvement")
    st.caption("🤝 Voting Ensemble (Top-3)")
    st.caption("📋 Cluster Profiling")
    st.divider()
    st.caption("LLM: Llama 3.3 70B via Groq")
    st.caption("Built by Anish")


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — RUN EXPERIMENT
# ══════════════════════════════════════════════════════════════════
if "Run" in page:

    st.title("🧪 Automated AI Scientist v3.0")
    st.markdown(
        "*Health → Hypothesis → Feature Eng → Train R1 → "
        "SHAP → Self-Improve R2 → Insight*"
    )
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📂 Dataset")
        st.caption(
            "**Supervised:** CSV needs a `target` column.  "
            "**Unsupervised:** any CSV or use keywords like *cluster / anomaly*."
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            os.makedirs(os.path.join(AI_DIR, "data"), exist_ok=True)
            sp = os.path.join(AI_DIR, "data", "sample.csv")
            with open(sp, "wb") as f:
                f.write(uploaded.getbuffer())
            dfp = pd.read_csv(sp)
            if "target" in dfp.columns:
                st.success(f"✅ {dfp.shape[0]} rows × {dfp.shape[1]} cols — supervised ready")
            else:
                st.info(f"ℹ️ {dfp.shape[0]} rows × {dfp.shape[1]} cols — unsupervised mode")
            with st.expander("Preview dataset"):
                st.dataframe(dfp.head(8), use_container_width=True)

        st.subheader("🧠 Experiment Instruction")
        user_prompt = st.text_area(
            "Tell the AI what to do",
            placeholder=(
                "Supervised:\n"
                "  • Try random forest and xgboost\n"
                "  • Compare all boosting models\n\n"
                "Unsupervised:\n"
                "  • Find clusters in this data\n"
                "  • Detect anomalies"
            ),
            height=130,
        )

        c1, c2 = st.columns(2)
        with c1:
            n_trials = st.slider("Optuna trials per model", 10, 50, 25, 5)
        with c2:
            st.markdown("**v3.0 Options**")
            enable_fe = st.checkbox("🔧 Feature Engineering", value=True,
                                     help="LLM invents new features before training")
            enable_si = st.checkbox("🔄 Self-Improvement", value=True,
                                     help="Auto-runs improved Round 2")

        run_btn = st.button("🚀 Run AI Scientist v3.0",
                            type="primary", use_container_width=True)

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

        lines = []
        def upd(stage, detail=""):
            lines.append(f"**{stage}** {detail}")
            status_box.markdown("\n\n".join(lines))

        with st.spinner("Running v3.0 pipeline..."):
            out = run_ai_scientist(
                user_prompt=user_prompt,
                progress_callback=upd,
                n_trials=n_trials,
                enable_feature_eng=enable_fe,
                enable_self_improve=enable_si,
            )

        status_box.success("✅ v3.0 pipeline complete!")
        st.divider()

        # ── Unpack ────────────────────────────────────────────────
        results    = out["results"]
        r2         = out.get("results_r2")
        rc         = out.get("round_comparison", {})
        ip         = out.get("improvement_plan", {})
        r2c        = out.get("round2_config", {})
        insight    = out["insight"]
        hyp        = out.get("hypothesis", "")
        health     = out.get("health", {})
        shap_data  = out.get("shap_result", {})
        fe         = out.get("feature_eng_result", {"new_features": [], "n_added": 0})
        rp         = out["report_path"]
        sel        = out["selected_models"]
        mode       = out.get("mode", "supervised")
        shape      = results.get("dataset_shape", [])
        code       = results.get("final_code", "")

        # ── Health Banner ─────────────────────────────────────────
        if health:
            score  = health.get("score", 100)
            grade  = health.get("grade", "?")
            issues = health.get("issues", [])
            with st.expander(
                f"🩺 Dataset Health: Grade **{grade}** ({score}/100) — {health.get('summary','')}",
                expanded=(score < 75)
            ):
                s = health.get("stats", {})
                h1, h2, h3, h4 = st.columns(4)
                with h1: st.metric("Rows",       s.get("rows", "?"))
                with h2: st.metric("Missing",    f"{s.get('missing_pct', 0)}%")
                with h3: st.metric("Duplicates", s.get("duplicate_rows", 0))
                with h4: st.metric("Memory MB",  s.get("memory_mb", 0))
                for i in issues:
                    if   i["severity"] == "high":   st.error(f"**{i['type'].replace('_',' ').title()}** — {i['message']}")
                    elif i["severity"] == "medium": st.warning(f"**{i['type'].replace('_',' ').title()}** — {i['message']}")
                    else:                            st.info(f"**{i['type'].replace('_',' ').title()}** — {i['message']}")
            st.divider()

        # ── Feature Eng Banner ────────────────────────────────────
        if fe.get("n_added", 0) > 0:
            added = [f["name"] for f in fe.get("new_features", []) if f["status"] == "added"]
            st.markdown(
                f'<div class="c-purple">🔧 <strong>Feature Engineering</strong> — '
                f'{fe["n_added"]} new features: ' + ", ".join(f"`{n}`" for n in added[:6]) + "</div>",
                unsafe_allow_html=True,
            )

        # ── Self-Improve Banner ───────────────────────────────────
        if rc:
            r1b   = rc.get("round1_best", {})
            r2b   = rc.get("round2_best", {})
            delta = rc.get("delta", 0)
            imp   = rc.get("improved", False)
            cls   = "c-green" if imp else "c-red"
            icon  = "✅" if imp else "⚠️"
            st.markdown(
                f'<div class="{cls}">'
                f'{icon} <strong>Self-Improvement Loop</strong> — '
                f'Round 1: <strong>{r1b.get("name","?")} ({r1b.get("score",0):.4f})</strong> → '
                f'Round 2: <strong>{r2b.get("name","?")} ({r2b.get("score",0):.4f})</strong> '
                f'Δ = <strong>{delta:+.4f}</strong></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ══════════════════════════════════════════════════════════
        # SUPERVISED — 8 TABS
        # ══════════════════════════════════════════════════════════
        if mode == "supervised":
            task   = results.get("task", "unknown")
            metric = "Accuracy" if task == "classification" else "RMSE"
            mdata  = [m for m in results.get("models", []) if m.get("score") is not None]
            ens    = results.get("ensemble", {})

            # Summary row
            st.subheader("📊 Results Summary")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.metric("Mode",   "SUPERVISED")
            with c2: st.metric("Task",   task.upper())
            with c3:
                if mdata:
                    b = mdata[0]
                    st.metric(f"Best {metric}", f"{b['score']:.4f}", delta=b["name"])
            with c4:
                if ens and not ens.get("error"):
                    st.metric("Ensemble", f"{ens['cv_score']:.4f}", delta="Top-3")
            with c5:
                st.metric("Dataset", f"{shape[0]}×{shape[1]}" if shape else "N/A")

            st.divider()

            t1,t2,t3,t4,t5,t6,t7,t8 = st.tabs([
                "🏆 Leaderboard",
                "🔄 Self-Improve",
                "🔧 Features",
                "⚙️ Parameters",
                "🔍 SHAP",
                "🔬 AI Insights",
                "💻 Final Code",
                "📄 Report",
            ])

            # ── Tab 1: Leaderboard ────────────────────────────────
            with t1:
                st.subheader(f"🏆 {metric} Leaderboard (5-fold CV)")
                st.caption("Pipeline: PolynomialFeatures → SelectKBest(30%) → Model")
                if mdata:
                    best = mdata[0]
                    st.markdown(
                        f'<div class="c-blue">🥇 <strong>{best["name"]}</strong>'
                        f' — {metric}: <strong>{best["score"]:.4f}</strong></div>',
                        unsafe_allow_html=True,
                    )
                    rows = []
                    for i, m in enumerate(mdata):
                        medal = ["🥇","🥈","🥉"][i] if i < 3 else f"#{i+1}"
                        rows.append({"Rank": medal, "Model": m["name"],
                                     f"{metric} (CV-5)": round(m["score"], 4),
                                     "Trials": m.get("n_trials", 25)})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    if ens and not ens.get("error"):
                        st.markdown(
                            f'<div class="c-yellow">🤝 <strong>Ensemble (Top-3 Voting)</strong>'
                            f' — {metric}: <strong>{ens["cv_score"]:.4f}</strong>'
                            f' | {", ".join(ens.get("models_used", []))}</div>',
                            unsafe_allow_html=True,
                        )
                    cdf = pd.DataFrame({"Model": [m["name"] for m in mdata],
                                        metric:  [m["score"] for m in mdata]}).set_index("Model")
                    st.bar_chart(cdf)
                else:
                    st.warning("No successful model results.")

            # ── Tab 2: Self-Improvement ───────────────────────────
            with t2:
                st.subheader("🔄 Iterative Self-Improvement Loop")
                st.caption("After Round 1, LLM analyzed results and ran an improved Round 2.")
                if rc:
                    r1b = rc["round1_best"]; r2b = rc["round2_best"]
                    delta = rc.get("delta", 0); imp = rc.get("improved", False)
                    ca, cb, cc = st.columns(3)
                    with ca: st.metric("Round 1 Best", f"{r1b['score']:.4f}", delta=r1b["name"])
                    with cb: st.metric("Round 2 Best", f"{r2b['score']:.4f}", delta=r2b["name"])
                    with cc: st.metric("Δ Improvement", f"{delta:+.4f}",
                                       delta=f"{rc.get('pct_change',0):.2f}% {'✅' if imp else '⚠️'}")
                    st.divider()
                    if ip:
                        with st.expander("🧠 LLM Improvement Plan", expanded=True):
                            if ip.get("reasoning"):
                                st.markdown(f"**Reasoning:** {ip['reasoning']}")
                            if ip.get("expected_improvement"):
                                st.markdown(f"**Expected outcome:** {ip['expected_improvement']}")
                    if r2c.get("changes_applied"):
                        st.markdown("**Changes applied for Round 2:**")
                        for ch in r2c["changes_applied"]:
                            st.markdown(f"- {ch}")
                    if r2:
                        r2_valid = [m for m in r2.get("models", []) if m.get("score") is not None]
                        if r2_valid:
                            st.subheader("Round 2 Leaderboard")
                            r2_rows = [{"Rank": ["🥇","🥈","🥉"][i] if i<3 else f"#{i+1}",
                                        "Model": m["name"],
                                        f"{metric} (CV-5)": round(m["score"],4)}
                                       for i, m in enumerate(r2_valid)]
                            st.dataframe(pd.DataFrame(r2_rows), use_container_width=True, hide_index=True)
                elif not enable_si:
                    st.info("Self-Improvement was disabled for this run.")
                else:
                    st.info("Self-Improvement did not produce results.")

            # ── Tab 3: Feature Engineering ────────────────────────
            with t3:
                st.subheader("🔧 LLM Feature Engineering")
                st.caption("LLM analyzed column names + correlations, invented new features as Python expressions.")
                nf = fe.get("new_features", [])
                if nf:
                    st.metric("New features created", fe.get("n_added", 0))
                    feat_rows = [{"Name": f["name"], "Expression": f["expression"],
                                  "Rationale": f["rationale"], "Status": f["status"]}
                                 for f in nf]
                    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)
                    st.caption("Failed expressions had syntax errors or missing columns — safely skipped.")
                elif not enable_fe:
                    st.info("Feature Engineering was disabled for this run.")
                else:
                    st.info("No valid features generated for this dataset.")

            # ── Tab 4: Parameters ─────────────────────────────────
            with t4:
                st.subheader("⚙️ Hyperparameter Results")
                st.caption(f"Optuna TPE + MedianPruner | {results.get('n_trials_per_model',25)} trials per model")
                for i, m in enumerate(mdata):
                    medal = ["🥇","🥈","🥉"][i] if i < 3 else f"#{i+1}"
                    with st.expander(f"{medal} {m['name']}  |  {metric}: {m.get('score',0):.4f}",
                                     expanded=(i == 0)):
                        ca, cb = st.columns(2)
                        with ca:
                            st.markdown(f"**Model:** `{m['name']}`")
                            st.markdown(f"**{metric} (CV-5):** `{m.get('score',0):.4f}`")
                            st.markdown(f"**Trials:** `{m.get('n_trials',25)}`")
                            st.markdown("**Pipeline:** PolyFeatures → SelectKBest → Model")
                        with cb:
                            params = m.get("best_params", {})
                            if params:
                                st.dataframe(
                                    pd.DataFrame([{"Parameter": k, "Best Value": str(v),
                                                   "Type": type(v).__name__}
                                                  for k, v in params.items()]),
                                    use_container_width=True, hide_index=True,
                                )
                            else:
                                st.info("No tunable hyperparameters")
                if ens and not ens.get("error"):
                    st.divider()
                    st.subheader("🤝 Ensemble Details")
                    st.markdown(f"**Models:** {', '.join(ens.get('models_used',[]))}")
                    st.markdown(f"**Voting:** `{ens.get('voting','hard')}`")
                    st.markdown(f"**CV {metric}:** `{ens.get('cv_score','N/A')}`")
                failed = [m for m in results.get("models", []) if m.get("score") is None]
                if failed:
                    st.divider()
                    st.markdown("**Failed Models:**")
                    for m in failed:
                        st.error(f"{m['name']}: {m.get('error','Unknown error')}")

            # ── Tab 5: SHAP ───────────────────────────────────────
            with t5:
                st.subheader("🔍 SHAP Feature Importance")
                st.caption(
                    "SHapley Additive exPlanations — shows which features drove the best model's predictions. "
                    "Mean |SHAP value| per feature."
                )
                if shap_data and not shap_data.get("error"):
                    top = shap_data.get("top_features", [])
                    if top:
                        st.markdown(
                            f"**Model explained:** `{shap_data.get('model_name','?')}` &nbsp;|&nbsp; "
                            f"**Explainer:** `{shap_data.get('explainer_type','?').title()}Explainer`"
                        )
                        sdf = pd.DataFrame(top).rename(columns={
                            "feature": "Feature", "importance": "Mean |SHAP|", "rank": "Rank"
                        })
                        sdf["Mean |SHAP|"] = sdf["Mean |SHAP|"].round(5)
                        st.dataframe(sdf[["Rank","Feature","Mean |SHAP|"]],
                                     use_container_width=True, hide_index=True)
                        cdf = pd.DataFrame({
                            "Feature":    [t["feature"]    for t in top],
                            "Mean |SHAP|": [t["importance"] for t in top],
                        }).set_index("Feature")
                        st.bar_chart(cdf)
                        st.caption(
                            "Higher = more important. A SHAP of 0.05 means the feature "
                            "shifts prediction by 0.05 units on average."
                        )
                elif shap_data.get("error"):
                    st.warning(f"SHAP not available: {shap_data['error']}")
                    st.code("pip install shap")
                else:
                    st.info("Install SHAP for feature importance: `pip install shap`")

            # ── Tab 6: AI Insights ────────────────────────────────
            with t6:
                st.subheader("🔬 Scientific Analysis by AI Researcher Agent")
                if hyp:
                    with st.expander("💡 Pre-Experiment Hypothesis (generated BEFORE training)",
                                     expanded=True):
                        st.markdown(f"*{hyp}*")
                st.info(insight)
                if rc:
                    with st.expander("🔄 Self-Improvement Commentary"):
                        if ip.get("reasoning"):
                            st.markdown(f"**Why changes were made:** {ip['reasoning']}")
                        if ip.get("expected_improvement"):
                            st.markdown(f"**Expected outcome:** {ip['expected_improvement']}")

            # ── Tab 7: Final Code ─────────────────────────────────
            with t7:
                st.subheader("💻 Final ML Code")
                st.caption("Pipeline (PolyFeatures → SelectKBest → Model) + Ensemble. Fully runnable.")
                if code:
                    st.code(code, language="python")
                    st.download_button("⬇️ Download final_ml_code.py",
                                       data=code, file_name="final_ml_code.py",
                                       mime="text/x-python",
                                       use_container_width=True, type="primary")
                else:
                    st.warning("No code generated.")

            # ── Tab 8: Report + Logs ──────────────────────────────
            with t8:
                st.subheader("📄 Download PDF Report")
                if rp and os.path.exists(rp):
                    with open(rp, "rb") as f:
                        fb = f.read()
                    ext  = os.path.splitext(rp)[1]
                    mime = "application/pdf" if ext == ".pdf" else "text/plain"
                    st.download_button(f"⬇️ Download Report ({ext.upper()})",
                                       data=fb, file_name=f"report{ext}",
                                       mime=mime, use_container_width=True)
                else:
                    st.info("Report not generated yet.")
                st.divider()
                st.subheader("📋 Agent Logs")
                st.text_area("Full pipeline logs",
                             "\n".join(out.get("logs", [])), height=350)

        # ══════════════════════════════════════════════════════════
        # UNSUPERVISED — 6 TABS
        # ══════════════════════════════════════════════════════════
        else:
            all_cls       = results.get("clustering", [])
            anom_names    = {"isolation forest","local outlier factor","lof","anomaly detection"}
            clustering    = [c for c in all_cls
                             if c.get("silhouette") is not None
                             and c["name"].lower() not in anom_names]
            anomalies     = [c for c in all_cls
                             if c.get("silhouette") is not None
                             and c["name"].lower() in anom_names]
            pca_coords    = results.get("pca_coords", [])
            best_labels   = results.get("best_labels", [])
            all_valid     = [c for c in all_cls if c.get("silhouette") is not None]
            best_overall  = max(all_valid, key=lambda x: x["silhouette"]) if all_valid else None

            # Summary
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Mode",          "UNSUPERVISED")
            with c2: st.metric("Algorithms Run", len(sel))
            with c3:
                if best_overall:
                    st.metric("Best Silhouette",
                              f"{best_overall['silhouette']:.4f}",
                              delta=best_overall["name"])
            with c4:
                st.metric("Dataset", f"{shape[0]}×{shape[1]}" if shape else "N/A")
            st.divider()

            ut1,ut2,ut3,ut4,ut5,ut6 = st.tabs([
                "🔵 Clusters","🔴 Anomalies","📋 Profiles",
                "🔬 AI Insights","💻 Final Code","📄 Report",
            ])

            with ut1:
                st.subheader("🔵 Clustering Leaderboard")
                st.caption("Composite: 0.5×Silhouette − 0.3×Davies-Bouldin + 0.2×Calinski-Harabasz")
                if clustering:
                    bc = clustering[0]
                    st.markdown(
                        f'<div class="c-green">🥇 <strong>{bc["name"]}</strong>'
                        f' — Silhouette: <strong>{bc["silhouette"]:.4f}</strong>'
                        f' | Clusters: <strong>{bc.get("n_clusters_found","?")}</strong></div>',
                        unsafe_allow_html=True,
                    )
                    rows = [{"Rank": ["🥇","🥈","🥉"][i] if i<3 else f"#{i+1}",
                             "Algorithm": c["name"],
                             "Silhouette ↑":        round(c.get("silhouette") or 0, 4),
                             "Davies-Bouldin ↓":    round(c.get("davies_bouldin") or 0, 4),
                             "Calinski-Harabasz ↑": round(c.get("calinski_harabasz") or 0, 1),
                             "Clusters Found":      c.get("n_clusters_found","?"),
                             "Noise Pts":           c.get("n_noise_points", 0)}
                            for i, c in enumerate(clustering)]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    if pca_coords and best_labels:
                        st.subheader("🗺️ PCA Cluster Visualization")
                        st.caption(f"2D PCA — colored by {bc['name']} assignments")
                        pca_df = pd.DataFrame(pca_coords, columns=["PC1","PC2"])
                        pca_df["Cluster"] = [str(l) for l in best_labels]
                        try:
                            import altair as alt
                            chart = (alt.Chart(pca_df)
                                     .mark_circle(size=60, opacity=0.7)
                                     .encode(x=alt.X("PC1:Q"), y=alt.Y("PC2:Q"),
                                             color=alt.Color("Cluster:N"),
                                             tooltip=["PC1","PC2","Cluster"])
                                     .properties(width="container", height=420)
                                     .interactive())
                            st.altair_chart(chart, use_container_width=True)
                        except ImportError:
                            st.scatter_chart(pca_df, x="PC1", y="PC2", color="Cluster")
                        pv = results.get("pca_variance", [])
                        if pv:
                            st.caption(f"PCA: {round(sum(pv)*100,1)}% total | "
                                       f"PC1: {round(pv[0]*100,1)}% | "
                                       f"PC2: {round(pv[1]*100,1) if len(pv)>1 else 'N/A'}%")

                    st.subheader("⚙️ Hyperparameters per Algorithm")
                    for i, c in enumerate(clustering):
                        medal = ["🥇","🥈","🥉"][i] if i<3 else f"#{i+1}"
                        with st.expander(
                            f"{medal} {c['name']}  |  Sil: {c.get('silhouette',0):.4f}"
                            f"  |  Clusters: {c.get('n_clusters_found','?')}",
                            expanded=(i == 0),
                        ):
                            ca, cb = st.columns(2)
                            with ca:
                                st.markdown(f"**Silhouette:** `{c.get('silhouette','N/A')}` *(max 1.0)*")
                                st.markdown(f"**Davies-Bouldin:** `{c.get('davies_bouldin','N/A')}` *(lower=better)*")
                                st.markdown(f"**Calinski-Harabasz:** `{c.get('calinski_harabasz','N/A')}` *(higher=better)*")
                                st.markdown(f"**Clusters found:** `{c.get('n_clusters_found','?')}`")
                                st.markdown(f"**Noise points:** `{c.get('n_noise_points',0)}`")
                            with cb:
                                params = {k:v for k,v in c.get("best_params",{}).items()
                                          if k not in ("anomalies_found","outliers_found")}
                                if params:
                                    st.dataframe(
                                        pd.DataFrame([{"Parameter":k,"Best Value":str(v)}
                                                      for k,v in params.items()]),
                                        use_container_width=True, hide_index=True,
                                    )
                failed_cls = [c for c in all_cls if c.get("silhouette") is None]
                if failed_cls:
                    st.divider()
                    for c in failed_cls:
                        st.error(f"{c['name']}: {c.get('error','Unknown error')}")

            with ut2:
                st.subheader("🔴 Anomaly Detection Results")
                st.caption("Labels: 0 = Normal, 1 = Anomaly")
                if anomalies:
                    for a in anomalies:
                        params  = a.get("best_params", {})
                        n_anom  = params.get("anomalies_found") or params.get("outliers_found","?")
                        n_total = shape[0] if shape else "?"
                        sil     = a.get("silhouette")
                        st.markdown(
                            f'<div class="c-red">🔴 <strong>{a["name"]}</strong>'
                            f' — Anomalies: <strong>{n_anom}</strong> / {n_total}'
                            f' | Silhouette: <strong>{f"{sil:.4f}" if sil else "N/A"}</strong></div>',
                            unsafe_allow_html=True,
                        )
                        dp = {k:v for k,v in params.items()
                              if k not in ("anomalies_found","outliers_found")}
                        if dp:
                            st.dataframe(
                                pd.DataFrame([{"Parameter":k,"Best Value":str(v)}
                                              for k,v in dp.items()]),
                                use_container_width=True, hide_index=True,
                            )
                        if pca_coords and best_labels:
                            st.caption(f"PCA scatter — anomalies vs normal ({a['name']})")
                            adf = pd.DataFrame(pca_coords, columns=["PC1","PC2"])
                            adf["Type"] = ["Anomaly" if l==1 else "Normal" for l in best_labels]
                            try:
                                import altair as alt
                                chart = (alt.Chart(adf).mark_circle(size=50, opacity=0.7)
                                         .encode(x="PC1:Q", y="PC2:Q",
                                                 color=alt.Color("Type:N", scale=alt.Scale(
                                                     domain=["Normal","Anomaly"],
                                                     range=["#3b82f6","#ef4444"])),
                                                 tooltip=["PC1","PC2","Type"])
                                         .properties(width="container", height=360)
                                         .interactive())
                                st.altair_chart(chart, use_container_width=True)
                            except ImportError:
                                adf["Color"] = [1 if t=="Anomaly" else 0 for t in adf["Type"]]
                                st.scatter_chart(adf, x="PC1", y="PC2", color="Color")
                        st.divider()
                else:
                    st.info("No anomaly detection ran. Add 'Isolation Forest' or 'Local Outlier Factor' to your prompt.")

            with ut3:
                st.subheader("📋 Cluster Profiles")
                st.caption("Defining features = highest z-score deviation from global mean.")
                cp = results.get("cluster_profile", {})
                if cp and cp.get("clusters"):
                    for cl in cp["clusters"]:
                        with st.expander(
                            f"🔵 Cluster {cl['label']} — {cl['size']} rows ({cl['pct']}%)",
                            expanded=(cl["label"] == 0),
                        ):
                            p1, p2 = st.columns(2)
                            with p1:
                                defs = cl.get("defining_features", [])
                                if defs:
                                    st.markdown("**Top defining features:**")
                                    st.dataframe(
                                        pd.DataFrame([{"Feature": d["feature"],
                                                       "Cluster Mean": d["cluster_mean"],
                                                       "Global Mean":  d["global_mean"],
                                                       "Z-Score":      d["z_score"],
                                                       "Direction":    d["direction"]}
                                                      for d in defs]),
                                        use_container_width=True, hide_index=True,
                                    )
                            with p2:
                                cm = cl.get("categorical_modes", {})
                                if cm:
                                    st.markdown("**Categorical modes:**")
                                    st.dataframe(
                                        pd.DataFrame([{"Feature":k,"Most Common":v}
                                                      for k,v in cm.items()]),
                                        use_container_width=True, hide_index=True,
                                    )
                    fs = cp.get("feature_summary", {})
                    if fs:
                        st.subheader("📊 Feature Means per Cluster")
                        try:
                            st.dataframe(
                                pd.DataFrame(fs).T.style.format("{:.3f}", na_rep="N/A"),
                                use_container_width=True,
                            )
                        except Exception:
                            st.dataframe(pd.DataFrame(fs).T, use_container_width=True)
                else:
                    st.info("Cluster profiles not available for this run.")

            with ut4:
                st.subheader("🔬 Scientific Analysis")
                st.info(insight)

            with ut5:
                st.subheader("💻 Final Unsupervised Code")
                st.caption("Runnable Python with StandardScaler, PCA, silhouette scoring.")
                if code:
                    st.code(code, language="python")
                    st.download_button("⬇️ Download final_unsupervised_code.py",
                                       data=code, file_name="final_unsupervised_code.py",
                                       mime="text/x-python",
                                       use_container_width=True, type="primary")
                else:
                    st.warning("No code generated.")

            with ut6:
                st.subheader("📄 Report")
                if rp and os.path.exists(rp):
                    with open(rp,"rb") as f: fb = f.read()
                    ext  = os.path.splitext(rp)[1]
                    mime = "application/pdf" if ext == ".pdf" else "text/plain"
                    st.download_button(f"⬇️ Download ({ext.upper()})",
                                       data=fb, file_name=f"report{ext}",
                                       mime=mime, use_container_width=True)
                else:
                    st.info("No report yet.")
                st.divider()
                st.subheader("📋 Agent Logs")
                st.text_area("Logs", "\n".join(out.get("logs",[])), height=300)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — LAB NOTEBOOK
# ══════════════════════════════════════════════════════════════════
elif "Notebook" in page:

    st.title("📓 Lab Notebook")
    st.markdown("*Every experiment — supervised & unsupervised, all rounds — stored in SQLite.*")
    st.divider()

    experiments = get_all_experiments()

    if not experiments:
        st.info("No experiments recorded yet. Run your first experiment!")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Experiments", len(experiments))
        with c2: st.metric("Supervised",   sum(1 for e in experiments if e.get("mode","supervised")=="supervised"))
        with c3: st.metric("Unsupervised", sum(1 for e in experiments if e.get("mode")=="unsupervised"))
        with c4: st.metric("Classification",sum(1 for e in experiments if e.get("task")=="classification"))
        st.divider()

        for exp in experiments:
            mode = exp.get("mode","supervised")
            task = exp.get("task","unknown")

            # Safe score conversion (fixes ValueError: format code 'f' for str)
            try:
                score_str = f"{float(exp['best_score']):.4f}" if exp.get("best_score") is not None else "N/A"
            except (TypeError, ValueError):
                score_str = "N/A"

            label = exp["user_prompt"][:60] + ("..." if len(exp["user_prompt"]) > 60 else "")

            if mode == "unsupervised":
                icon  = "🟢"
                badge = f"UNSUPERVISED | {exp['best_model']} (Sil: {score_str})"
            elif task == "classification":
                icon  = "🔵"
                badge = f"CLASSIFICATION | {exp['best_model']} (Acc: {score_str})"
            else:
                icon  = "🟠"
                badge = f"REGRESSION | {exp['best_model']} (RMSE: {score_str})"

            auto = " [auto-improved]" if "[v3 Round 2]" in exp.get("user_prompt","") else ""

            with st.expander(f"{icon} [{exp['timestamp']}]  {label}{auto}  →  {badge}"):
                col1, col2 = st.columns(2)
                er = exp.get("results", {})

                with col1:
                    st.markdown(f"**Prompt:** {exp['user_prompt']}")
                    st.markdown(f"**Mode:** `{mode.upper()}`")
                    st.markdown(f"**Task:** `{task.upper()}`")
                    st.markdown(f"**Dataset:** {exp.get('dataset_shape','N/A')}")
                    st.markdown(f"**Algorithms:** {', '.join(exp.get('selected_models',[]))}")
                    ens_nb = er.get("ensemble", {})
                    if ens_nb and not ens_nb.get("error"):
                        ml = "Accuracy" if task == "classification" else "RMSE"
                        st.markdown(f"**🤝 Ensemble {ml}:** `{ens_nb.get('cv_score','N/A')}`")

                with col2:
                    if mode == "unsupervised":
                        cd = [c for c in er.get("clustering",[]) if c.get("silhouette") is not None]
                        if cd:
                            rows = []
                            for c in cd:
                                p = c.get("best_params", {})
                                n_a = p.get("anomalies_found") or p.get("outliers_found")
                                row = {"Algorithm":      c["name"],
                                       "Silhouette ↑":   round(c.get("silhouette") or 0, 4),
                                       "Clusters Found": c.get("n_clusters_found","?"),
                                       "Noise Points":   c.get("n_noise_points", 0)}
                                if n_a is not None:
                                    row["Anomalies"] = n_a
                                rows.append(row)
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        ml   = "Accuracy" if task == "classification" else "RMSE"
                        mdat = [m for m in er.get("models",[]) if m.get("score") is not None]
                        if mdat:
                            st.dataframe(
                                pd.DataFrame([{"Model": m["name"],
                                               f"{ml} (CV-5)": round(m["score"],4)}
                                              for m in mdat]),
                                use_container_width=True, hide_index=True,
                            )

                if exp.get("insight"):
                    st.markdown("**🔬 AI Insight:**")
                    st.info(exp["insight"])

                fc = er.get("final_code","")
                if fc:
                    fname = (f"{'unsupervised' if mode=='unsupervised' else 'ml'}"
                             f"_code_exp_{exp['id']}.py")
                    st.download_button("⬇️ Download Code", data=fc,
                                       file_name=fname, mime="text/x-python",
                                       key=f"dl_{exp['id']}")

        st.divider()
        if st.button("🗑️ Clear All Experiments", type="secondary"):
            clear_all_experiments()
            st.rerun()
