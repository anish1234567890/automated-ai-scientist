# ui — Streamlit Dashboard

This folder contains the Streamlit frontend for the Automated AI Scientist.

## Run

Make sure you've set up the backend first (see `../ai_scientist/README.md`).

```bash
cd ui
streamlit run streamlit_app.py
```

The dashboard will open at `http://localhost:8501`

## Features

| Tab | What it shows |
|---|---|
| 🏆 Leaderboard | Ranked model scores + bar chart |
| ⚙️ Parameters | Optuna best hyperparameters per model |
| 🔬 AI Insights | LLM scientific analysis |
| 💻 Final Code | Downloadable Python with best params |
| 📄 Report | PDF download + agent logs |
| 📓 Lab Notebook | Full history of all experiments |

## Notes

- Upload your CSV via the UI — no need to copy files manually
- The dataset must have a column named `target`
- Adjust Optuna trials (10–50) using the slider — more trials = better tuning but slower