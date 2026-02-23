import streamlit as st
import sys
import os

# connect backend
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ai_scientist")
sys.path.append(os.path.abspath(backend_path))

from app import run_ai_scientist

st.set_page_config(page_title="AI Scientist", layout="wide")

st.title("🧪 Automated AI Scientist")
st.write("Multi-Agent AI Research System")

if st.button("Run AI Scientist"):
    with st.spinner("Running experiments..."):
        logs = run_ai_scientist()

    st.success("Experiment finished")

    st.subheader("Agent Logs")
    st.text_area("Logs", logs, height=400)
