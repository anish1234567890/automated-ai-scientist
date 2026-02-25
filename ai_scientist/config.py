import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "sample.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
GENERATED_CODE_PATH = os.path.join(OUTPUT_DIR, "generated_script.py")
RESULT_PATH = os.path.join(OUTPUT_DIR, "results.json")
DB_PATH = os.path.join(OUTPUT_DIR, "lab_notebook.db")
REPORT_PATH = os.path.join(OUTPUT_DIR, "report.pdf")
LOGS_PATH = os.path.join(OUTPUT_DIR, "logs.txt")

# Models
MODEL_CODER = "llama-3.3-70b-versatile"
MODEL_RESEARCHER = "llama-3.3-70b-versatile"

# System
MAX_RETRIES = 3
