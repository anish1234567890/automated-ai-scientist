# Automated AI Scientist

An autonomous multi-agent system that conducts machine learning experiments through iterative hypothesis generation, code implementation, execution, and scientific reflection. This project implements an AI-powered research assistant capable of autonomously designing, running, and evaluating ML experiments.

## 🎯 Project Overview

The **Automated AI Scientist** is a cutting-edge project for 2026, inspired by research from **Sakana AI** (AI Scientist system) and **Google Research** (AI Co-scientist). Unlike traditional chatbots, this system implements a **"write-execute-reflect"** loop where an LLM acts as a lead scientist, autonomously generating hypotheses, writing code, executing experiments, analyzing results, and iterating to improve outcomes.

### Core Objectives

- **Ideation:** Generate hypotheses (e.g., "Increasing the learning rate will speed up convergence but might lower final accuracy")
- **Implementation:** Automatically write Python scripts to test hypotheses
- **Execution:** Run scripts in a sandboxed environment and capture metrics (loss, accuracy, charts)
- **Evaluation:** Analyze results and autonomously decide the next experiment

## 🏗️ System Architecture

The system consists of four specialized agents working together:

| Component | Responsibility | Technical Tool |
|-----------|---------------|----------------|
| **Ideator Agent** | Brainstorms new hyperparameter combinations or model tweaks | Llama 3 / Mistral (via Ollama) |
| **Coder Agent** | Translates ideas into Python/Scikit-learn scripts | CodeLlama / DeepSeek-Coder |
| **Executor** | Runs code in a sandboxed environment | Python `subprocess` or Docker |
| **Reviewer Agent** | Critiques results and identifies failures | Gemini / Llama 3 |

## 🔄 The "AI Scientist" Workflow

The system follows an iterative experimental cycle:

1. **Seed Prompt:** User provides a dataset and a goal (e.g., "Find the best model for this CSV")
2. **Experiment Planning:** The LLM proposes multiple "trials" (e.g., SVM vs. Random Forest)
3. **The Sandbox Loop:**
   - **Attempt 1:** LLM generates code → Script runs → Script crashes (e.g., "module not found")
   - **Self-Correction:** Error message is fed back to the LLM → Code is fixed
   - **Attempt 2:** Script runs → Success → Results (e.g., 0.85 accuracy) are saved
4. **Scientific Reflection:** Agent analyzes results and generates insights (e.g., "The model is overfitting. For the next experiment, I will increase regularization")
5. **Final Report:** After N iterations, generates a comprehensive summary of findings

## 🛠️ Tech Stack

Designed for **Google Colab (T4 GPU)** and **Open Source LLMs**:

- **LLM Hosting:** Ollama or vLLM (to run Llama 3 or Mistral on T4 GPU)
- **Agent Framework:** LangGraph or CrewAI (manages loops and hand-offs between agents)
- **Database:** SQLite or JSON files (stores the "lab notebook" - history of all experiments)
- **Visualization:** Matplotlib/Seaborn (agent saves plots as `.png` files)
- **Frontend:** Streamlit (real-time dashboard showing AI "thinking" and plotting)

## ✨ Unique Features

To distinguish this project from basic automation scripts:

- **Automated Peer Review:** A second LLM instance acts as a "Critical Reviewer" that identifies flaws in the first agent's logic
- **Literature Search:** Integration with APIs like **Semantic Scholar** to search for real papers related to the dataset
- **Safety Sandbox:** A "Code Guard" that scans LLM-generated code for dangerous commands (like `rm -rf /`) before execution

## 📋 Implementation Roadmap

- **Week 1-2:** Set up Ollama on Colab and create a "Hello World" demo of an LLM writing and running a simple math script
- **Week 3-4:** Build the feedback loop - feed errors back to the LLM and ensure it can fix its own code
- **Week 5-6:** Connect the agent to a real ML task (e.g., Titanic or Iris dataset)
- **Week 7-8:** Build the Streamlit UI and final reporting module

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google Colab account (for T4 GPU access)
- Ollama or vLLM installed
- Required Python packages (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd automated-ai-scientist

# Install dependencies
pip install -r requirements.txt

# Set up Ollama (if using local LLM)
# Follow Ollama installation guide for your platform
```

### Usage

```python
# Example: Initialize the AI Scientist
from ai_scientist import AutomatedAIScientist

scientist = AutomatedAIScientist(
    dataset_path="data/iris.csv",
    goal="Find the best classification model"
)

# Run autonomous experiments
results = scientist.run_experiments(max_iterations=10)

# Generate final report
scientist.generate_report()
```

## 📁 Project Structure

```
automated-ai-scientist/
├── README.md
├── requirements.txt
├── src/
│   ├── agents/
│   │   ├── ideator.py
│   │   ├── coder.py
│   │   └── reviewer.py
│   ├── executor/
│   │   └── sandbox.py
│   ├── utils/
│   │   ├── code_guard.py
│   │   └── literature_search.py
│   └── main.py
├── data/
├── experiments/
│   └── lab_notebook.db
└── frontend/
    └── streamlit_app.py
```

## 🤝 Contributing

This is a major project for academic purposes. Contributions and suggestions are welcome!

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

- Inspired by **Sakana AI**'s AI Scientist system
- Inspired by **Google Research**'s AI Co-scientist
- Built with open-source LLMs and frameworks

## 📧 Contact

[Your contact information]

---

**Note:** This project is designed as a comprehensive demonstration of autonomous AI research capabilities, combining multiple LLM agents, code generation, execution, and scientific reasoning.
