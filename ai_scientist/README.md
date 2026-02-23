# AI Scientist Agent (Multi-LLM)

Run from the `ai_scientist` folder.

## Setup

```bash
cd ai_scientist
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your **GEMINI_API_KEY** and **GROQ_API_KEY**.

## Run

```bash
python app.py
```

## Dashboard

```bash
streamlit run streamlit_app.py
```

Put your dataset as `data.csv` in this folder for the generated code to use.
