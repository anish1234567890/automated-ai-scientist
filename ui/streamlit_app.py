import streamlit as st
import sys
import os
import json

sys.path.append(os.path.abspath("../ai_scientist"))
from app import run_ai_scientist

st.set_page_config(page_title="Automated AI Scientist", layout="wide")

st.title("🧪 Automated AI Scientist")
st.write("AutoML Multi-Agent System")

# ---------------- DATASET ----------------
st.subheader("📂 Upload Dataset")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    with open("../ai_scientist/data/sample.csv", "wb") as f:
        f.write(uploaded.getbuffer())
    st.success("Dataset uploaded")

# ---------------- USER MODEL INPUT ----------------
st.subheader("🧠 Experiment Instructions")
user_prompt = st.text_area(
    "Tell AI what models to try",
    placeholder="Example: try random forest, svm and deep learning"
)

# ---------------- RUN BUTTON ----------------
if st.button("🚀 Run AI Scientist"):

    if not user_prompt:
        st.warning("Please enter experiment instruction")
        st.stop()

    with st.spinner("Running AI experiments..."):
        logs = run_ai_scientist(user_prompt)

    st.success("Experiment Completed")

    # ---------------- LOGS ----------------
    st.subheader("🧠 Agent Logs")
    st.text_area("Logs", logs, height=300)

    # ---------------- GENERATED CODE ----------------
    st.subheader("💻 Generated ML Code")
    code_path = "../ai_scientist/outputs/generated_script.py"

    if os.path.exists(code_path):
        with open(code_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="python")
    else:
        st.info("No code generated yet")

    # ---------------- RESULTS ----------------
    results_path = "../ai_scientist/outputs/results.json"

    if os.path.exists(results_path):
        with open(results_path) as f:
            try:
                data = json.load(f)

                st.subheader("🏆 Model Leaderboard")

                if "models" in data:
                    for m in data["models"]:
                        st.write(f"**{m['name']}** → Accuracy: {m['accuracy']}")

                    # chart data
                    names = [m["name"] for m in data["models"]]
                    accs = [m["accuracy"] for m in data["models"]]

                    st.subheader("📈 Accuracy Comparison")
                    chart_data = {names[i]: accs[i] for i in range(len(names))}
                    st.bar_chart(chart_data)

            except:
                st.error("Error reading results.json")
    else:
        st.info("No results yet")

st.divider()
st.caption("Built by Anish | Automated AI Scientist")
