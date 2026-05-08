import streamlit as st
import sys
import os
import re
import time
import pandas as pd
import numpy as np

# ── Path setup ────────────────────────────────────────────────────
AI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai_scientist"))
if AI_DIR not in sys.path:
    sys.path.insert(0, AI_DIR)
os.chdir(AI_DIR)

from app import run_ai_scientist
from core.lab_notebook import get_all_experiments, clear_all_experiments
from config import DATA_PATH


def _fmt4(value):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"

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
            try:
                dfp = pd.read_csv(sp)
            except Exception as e:
                st.error(f"Could not read CSV: {e}. Please upload a valid UTF-8 CSV file.")
                st.stop()
            if "target" in dfp.columns:
                st.success(f"✅ {dfp.shape[0]} rows × {dfp.shape[1]} cols — supervised ready")
            else:
                st.info(f"ℹ️ {dfp.shape[0]} rows × {dfp.shape[1]} cols — unsupervised mode")
            with st.expander("Preview dataset"):
                st.dataframe(dfp.head(8), width="stretch")

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
                            type="primary", width="stretch")

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

        status_box.empty()
        with col2:
            round_header_ph = st.empty()
            overall_label_ph = st.empty()
            overall_bar_ph = st.empty()
            model_title_ph = st.empty()
            model_bar_ph = st.empty()
            model_meta_ph = st.empty()
            model_done_ph = st.empty()
            log_ph = st.empty()

        state = {
            "log_lines": [],
            "done_lines": [],
            "round": 1,
            "stats": {
                1: {"completed": 0, "total": 0},
                2: {"completed": 0, "total": 0},
            },
            "model_index": 0,
            "current_model": "",
            "current_trial": 0,
            "current_trial_total": 0,
            "current_best": None,
            "model_start_time": None,
        }

        def _clamp_progress(value):
            try:
                return max(0.0, min(float(value), 1.0))
            except Exception:
                return 0.0

        def _safe_markdown(ph, text):
            try:
                ph.markdown(text)
            except Exception:
                pass

        def _safe_caption(ph, text):
            try:
                ph.caption(text)
            except Exception:
                pass

        def _safe_progress(ph, value):
            try:
                ph.progress(_clamp_progress(value))
            except Exception:
                pass

        def _safe_empty(ph):
            try:
                ph.empty()
            except Exception:
                pass

        def _safe_success(ph, text):
            try:
                ph.success(text)
            except Exception:
                pass

        def _current_stats():
            return state["stats"][state["round"]]

        def _set_total(total_models):
            s = _current_stats()
            s["total"] = max(int(total_models), 0)
            if s["total"] > 0 and s["completed"] > s["total"]:
                s["completed"] = s["total"]

        def _increment_completed():
            s = _current_stats()
            if s["total"] > 0:
                s["completed"] = min(s["completed"] + 1, s["total"])
            else:
                s["completed"] += 1

        def _render_overall():
            s = _current_stats()
            prefix = "Overall" if state["round"] == 1 else "Round 2"
            if s["total"] > 0:
                _safe_markdown(
                    overall_label_ph,
                    f"**{prefix}: {s['completed']}/{s['total']} models complete**",
                )
                _safe_progress(overall_bar_ph, min(s["completed"] / s["total"], 1.0))
            else:
                _safe_markdown(overall_label_ph, f"**{prefix}: waiting for model tuning...**")
                _safe_progress(overall_bar_ph, 0.0)

        def _clear_model_panel():
            _safe_empty(model_title_ph)
            _safe_empty(model_bar_ph)
            _safe_empty(model_meta_ph)

        def _render_model_progress():
            s = _current_stats()
            if not state["current_model"]:
                _clear_model_panel()
                return

            _safe_markdown(
                model_title_ph,
                f"⚙️ Tuning **{state['current_model']}** "
                f"({state['model_index']}/{s['total']})",
            )
            if state["current_trial_total"] > 0:
                in_progress_value = min(
                    state["current_trial"] / state["current_trial_total"],
                    0.99,
                )
                _safe_progress(model_bar_ph, in_progress_value)
                best_txt = f"{state['current_best']:.4f}" if state["current_best"] is not None else "N/A"
                _safe_caption(
                    model_meta_ph,
                    f"Trial {state['current_trial']}/{state['current_trial_total']} • "
                    f"Best so far: {best_txt}",
                )
            else:
                _safe_progress(model_bar_ph, 0.0)
                _safe_caption(model_meta_ph, "Waiting for trial updates...")

        def _finalize_current_model(increment_counter=True):
            if not state["current_model"]:
                return
            elapsed = 0.0
            if state["model_start_time"] is not None:
                elapsed = time.time() - state["model_start_time"]
            trials = state["current_trial_total"] or state["current_trial"] or "?"
            score_txt = f"{state['current_best']:.4f}" if state["current_best"] is not None else "N/A"
            state["done_lines"].append(
                f"✅ {state['current_model']} — Score: {score_txt} ({trials} trials, {elapsed:.1f}s)"
            )
            _safe_markdown(model_done_ph, "\n\n".join(state["done_lines"][-10:]))
            if increment_counter:
                _increment_completed()
            state["current_model"] = ""
            state["current_trial"] = 0
            state["current_trial_total"] = 0
            state["current_best"] = None
            state["model_start_time"] = None
            _render_overall()
            _clear_model_panel()

        def _switch_to_round2():
            _finalize_current_model(increment_counter=True)
            state["round"] = 2
            state["stats"][2] = {"completed": 0, "total": 0}
            _safe_markdown(round_header_ph, "**🔄 Round 2 — Auto-Improved**")
            _clear_model_panel()
            _render_overall()

        _render_overall()

        def upd(stage, detail=""):
            stage_txt = str(stage).strip()
            detail_txt = str(detail).strip() if detail not in (None, "") else ""

            if "AutoML Round 2" in stage_txt:
                _switch_to_round2()

            model_match = re.search(r"Tuning\s*\[(\d+)\s*/\s*(\d+)\]\s*:\s*(.+)$", stage_txt)
            if model_match:
                idx = int(model_match.group(1))
                total = int(model_match.group(2))
                model_name = model_match.group(3).strip()

                if state["current_model"] and state["current_model"] != model_name:
                    _finalize_current_model(increment_counter=True)

                state["model_index"] = idx
                _set_total(total)
                state["current_model"] = model_name
                state["current_trial"] = 0
                state["current_trial_total"] = 0
                state["current_best"] = None
                state["model_start_time"] = time.time()
                _render_overall()
                _render_model_progress()
                return

            trial_match = re.search(r"trial\s+(\d+)\s*/\s*(\d+)", stage_txt, re.IGNORECASE)
            if trial_match and state["current_model"]:
                trial_num = int(trial_match.group(1))
                trial_total = int(trial_match.group(2))
                state["current_trial"] = trial_num
                state["current_trial_total"] = trial_total

                score_match = re.search(r"best so far[:\s]+([\d.]+)", detail_txt, re.IGNORECASE)
                if score_match:
                    try:
                        state["current_best"] = float(score_match.group(1))
                    except Exception:
                        pass

                if trial_num % 3 == 0 or trial_num == trial_total:
                    _render_model_progress()
                return

            if state["current_model"] and not stage_txt.lower().startswith("trial"):
                _finalize_current_model(increment_counter=True)

            line = stage_txt if not detail_txt else f"{stage_txt} {detail_txt}"
            if line:
                state["log_lines"].append(f"**{line}**")
                _safe_markdown(log_ph, "\n\n".join(state["log_lines"][-14:]))

        with st.spinner("Running v3.0 pipeline..."):
            out = run_ai_scientist(
                user_prompt=user_prompt,
                progress_callback=upd,
                n_trials=n_trials,
                enable_feature_eng=enable_fe,
                enable_self_improve=enable_si,
            )

        _finalize_current_model(increment_counter=True)
        s = _current_stats()
        if s["total"] > 0 and s["completed"] < s["total"]:
            s["completed"] = s["total"]
            _render_overall()
        _safe_success(overall_label_ph, "✅ v3.0 pipeline complete!")
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
                f'Round 1: <strong>{r1b.get("name","?")} ({_fmt4(r1b.get("score"))})</strong> → '
                f'Round 2: <strong>{r2b.get("name","?")} ({_fmt4(r2b.get("score"))})</strong> '
                f'Δ = <strong>{delta:+.4f}</strong></div>',
                unsafe_allow_html=True,
            )
            if delta > 0:
                st.success(f"✅ Round 2 improved by +{delta:.4f}")
            elif delta < 0:
                st.warning(f"⚠️ Round 2 did not improve ({delta:.4f})")
            else:
                st.info("➡️ No change between rounds")

        st.divider()

        # ══════════════════════════════════════════════════════════
        # SUPERVISED — 8 TABS
        # ══════════════════════════════════════════════════════════
        if mode == "supervised":
            task   = results.get("task", "unknown")
            metric = "Accuracy" if task == "classification" else "RMSE"
            mdata  = [m for m in results.get("models", []) if m.get("score") is not None]
            r2_data = [m for m in (r2 or {}).get("models", []) if m.get("score") is not None]
            if task == "classification":
                r2_data = sorted(r2_data, key=lambda x: x["score"], reverse=True)
            else:
                r2_data = sorted(r2_data, key=lambda x: x["score"])
            ens    = results.get("ensemble", {})
            r1_best = mdata[0] if mdata else {}
            r2_best = rc.get("round2_best", {}) if rc else {}
            round2_improved = bool(rc.get("improved")) and bool(r2_data)

            if round2_improved and r2_best:
                best_model = r2_best.get("name", r2_data[0]["name"])
                best_score = r2_best.get("score", r2_data[0]["score"])
                best_round = "Round 2"
                best_badge = "🔄 Round 2"
            else:
                best_model = r1_best.get("name", "N/A")
                best_score = r1_best.get("score")
                best_round = "Round 1"
                best_badge = "⚙️ Round 1"

            # Summary row
            st.subheader("📊 Results Summary")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.metric("Mode",   "SUPERVISED")
            with c2: st.metric("Task",   task.upper())
            with c3:
                if best_model != "N/A":
                    st.metric(f"Best {metric}", _fmt4(best_score), delta=best_model)
                    st.caption(best_badge)
            with c4:
                if ens and not ens.get("error"):
                    st.metric("Ensemble", _fmt4(ens.get("cv_score")), delta="Top-3")
            with c5:
                st.metric("Dataset", f"{shape[0]}×{shape[1]}" if shape else "N/A")

            if mdata:
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown(
                        f"**⚙️ Round 1:** {r1_best.get('name', 'N/A')} — {_fmt4(r1_best.get('score'))}"
                    )
                with s2:
                    if r2_data:
                        r2_best_name = r2_best.get("name", r2_data[0]["name"])
                        r2_best_score = r2_best.get("score", r2_data[0]["score"])
                        st.markdown(f"**🔄 Round 2:** {r2_best_name} — {_fmt4(r2_best_score)}")
                    else:
                        st.markdown("**🔄 Round 2:** N/A")
                with s3:
                    st.markdown(f"**Overall Best ({best_round}):** {best_model} — {_fmt4(best_score)}")

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
                    st.markdown(
                        f'<div class="c-blue">🥇 <strong>{best_model}</strong>'
                        f' — {metric}: <strong>{_fmt4(best_score)}</strong>'
                        f' &nbsp;|&nbsp; <strong>{best_badge}</strong></div>',
                        unsafe_allow_html=True,
                    )

                    if round2_improved:
                        st.markdown("### 🔄 Round 2 Results (Auto-Improved)")
                        r2_best_name = r2_best.get("name", r2_data[0]["name"])
                        r2_rows = []
                        for i, m in enumerate(r2_data):
                            medal = ["🥇","🥈","🥉"][i] if i < 3 else f"#{i+1}"
                            r2_rows.append({"Rank": medal, "Model": m["name"],
                                            f"{metric} (CV-5)": round(m["score"], 4),
                                            "Trials": m.get("n_trials", 25)})
                        r2_df = pd.DataFrame(r2_rows)
                        r2_style = r2_df.style.apply(
                            lambda row: ["background-color: #dcfce7" if row["Model"] == r2_best_name else ""
                                         for _ in row],
                            axis=1,
                        )
                        st.dataframe(r2_style, width="stretch", hide_index=True)

                        with st.expander("⚙️ Round 1 Results", expanded=False):
                            r1_rows = []
                            for i, m in enumerate(mdata):
                                medal = ["🥇","🥈","🥉"][i] if i < 3 else f"#{i+1}"
                                r1_rows.append({"Rank": medal, "Model": m["name"],
                                                f"{metric} (CV-5)": round(m["score"], 4),
                                                "Trials": m.get("n_trials", 25)})
                            st.dataframe(pd.DataFrame(r1_rows), width="stretch", hide_index=True)
                    else:
                        st.markdown("### ⚙️ Round 1 Results")
                        rows = []
                        for i, m in enumerate(mdata):
                            medal = ["🥇","🥈","🥉"][i] if i < 3 else f"#{i+1}"
                            rows.append({"Rank": medal, "Model": m["name"],
                                         f"{metric} (CV-5)": round(m["score"], 4),
                                         "Trials": m.get("n_trials", 25)})
                        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    if ens and not ens.get("error"):
                        st.markdown(
                            f'<div class="c-yellow">🤝 <strong>Ensemble (Top-3 Voting)</strong>'
                            f' — {metric}: <strong>{_fmt4(ens.get("cv_score"))}</strong>'
                            f' | {", ".join(ens.get("models_used", []))}</div>',
                            unsafe_allow_html=True,
                        )
                    chart_data = r2_data if round2_improved else mdata
                    cdf = pd.DataFrame({"Model": [m["name"] for m in chart_data],
                                        metric:  [m["score"] for m in chart_data]}).set_index("Model")
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
                    with ca: st.metric("Round 1 Best", _fmt4(r1b.get("score")), delta=r1b["name"])
                    with cb: st.metric("Round 2 Best", _fmt4(r2b.get("score")), delta=r2b["name"])
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
                            st.dataframe(pd.DataFrame(r2_rows), width="stretch", hide_index=True)
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
                    st.dataframe(pd.DataFrame(feat_rows), width="stretch", hide_index=True)
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
                    with st.expander(f"{medal} {m['name']}  |  {metric}: {_fmt4(m.get('score'))}",
                                     expanded=(i == 0)):
                        ca, cb = st.columns(2)
                        with ca:
                            st.markdown(f"**Model:** `{m['name']}`")
                            st.markdown(f"**{metric} (CV-5):** `{_fmt4(m.get('score'))}`")
                            st.markdown(f"**Trials:** `{m.get('n_trials',25)}`")
                            st.markdown("**Pipeline:** PolyFeatures → SelectKBest → Model")
                        with cb:
                            params = m.get("best_params", {})
                            if params:
                                st.dataframe(
                                    pd.DataFrame([{"Parameter": k, "Best Value": str(v),
                                                   "Type": type(v).__name__}
                                                  for k, v in params.items()]),
                                    width="stretch", hide_index=True,
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
                                     width="stretch", hide_index=True)
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
                                       width="stretch", type="primary")
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
                                       mime=mime, width="stretch")
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
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

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
                            st.altair_chart(chart, width="stretch")
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
                                        width="stretch", hide_index=True,
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
                                width="stretch", hide_index=True,
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
                                st.altair_chart(chart, width="stretch")
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
                                        width="stretch", hide_index=True,
                                    )
                            with p2:
                                cm = cl.get("categorical_modes", {})
                                if cm:
                                    st.markdown("**Categorical modes:**")
                                    st.dataframe(
                                        pd.DataFrame([{"Feature":k,"Most Common":v}
                                                      for k,v in cm.items()]),
                                        width="stretch", hide_index=True,
                                    )
                    fs = cp.get("feature_summary", {})
                    if fs:
                        st.subheader("📊 Feature Means per Cluster")
                        try:
                            st.dataframe(
                                pd.DataFrame(fs).T.style.format("{:.3f}", na_rep="N/A"),
                                width="stretch",
                            )
                        except Exception:
                            st.dataframe(pd.DataFrame(fs).T, width="stretch")
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
                                       width="stretch", type="primary")
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
                                       mime=mime, width="stretch")
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
                            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    else:
                        ml   = "Accuracy" if task == "classification" else "RMSE"
                        mdat = [m for m in er.get("models",[]) if m.get("score") is not None]
                        if mdat:
                            st.dataframe(
                                pd.DataFrame([{"Model": m["name"],
                                               f"{ml} (CV-5)": round(m["score"],4)}
                                              for m in mdat]),
                                width="stretch", hide_index=True,
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

