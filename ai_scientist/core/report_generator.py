import os
from datetime import datetime


def _get_report_path():
    from config import REPORT_PATH
    return REPORT_PATH


try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


class _PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(20, 20, 20)
        self.cell(0, 10, "Automated AI Scientist - Experiment Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(130, 130, 130)
        self.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()} | Automated AI Scientist", align="C")


    def section(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 70, 150)
        self.ln(3)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 70, 150)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_text_color(30, 30, 30)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def row(self, label, value, highlight=False):
        self.set_font("Helvetica", "B", 10)
        fill_color = (225, 240, 255) if highlight else (245, 245, 245)
        self.set_fill_color(*fill_color)
        self.cell(70, 7, label, border=1, fill=True)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 7, str(value), border=1, fill=highlight, new_x="LMARGIN", new_y="NEXT")


def _safe(text: str) -> str:
    """Strip characters outside Latin-1 range so Helvetica never crashes."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf_report(results: dict, insight: str, user_prompt: str, mode: str = 'supervised') -> str:
    """
    Generate a PDF experiment report.
    Falls back to a .txt file if fpdf2 is not installed.
    Returns the path of the saved file.
    """
    report_path = _get_report_path()
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    task   = results.get("task", "unknown")
    metric = "Accuracy" if task == "classification" else "RMSE"

    # ── Fallback: plain text ──────────────────────────────────────
    if not FPDF_AVAILABLE:
        txt_path = report_path.replace(".pdf", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("AUTOMATED AI SCIENTIST - EXPERIMENT REPORT\n")
            f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"User Prompt   : {user_prompt}\n")
            f.write(f"Mode          : {mode.upper()}\n")
            f.write(f"Dataset Shape : {results.get('dataset_shape', 'N/A')}\n\n")
            if mode == "unsupervised":
                f.write("CLUSTERING RESULTS:\n")
                for c in results.get("clustering", []):
                    sil = c.get("silhouette")
                    s   = f"{sil:.4f}" if sil is not None else "Failed"
                    f.write(f"  {c['name']:25s} Silhouette: {s}  Params: {c.get('best_params', {})}\n")
            else:
                f.write("MODEL RESULTS:\n")
                for m in results.get("models", []):
                    score = m.get("score")
                    s_str = f"{score:.4f}" if score is not None else "Failed"
                    f.write(f"  {m['name']:25s} {metric}: {s_str}  Params: {m.get('best_params', {})}\n")
            f.write(f"\nAI INSIGHTS:\n{insight}\n")
        return txt_path

    # ── Unsupervised PDF ──────────────────────────────────────────
    if mode == "unsupervised":
        pdf = _PDF()
        pdf.add_page()

        pdf.section("1. Experiment Overview")
        pdf.row("User Prompt",    _safe(user_prompt))
        pdf.row("Mode",           "UNSUPERVISED LEARNING")
        pdf.row("Dataset Shape",  str(results.get("dataset_shape", "N/A")))
        pdf.row("Trials / Algo",  str(results.get("n_trials_per_algo", 20)))
        pdf.row("PCA Variance",   str(results.get("pca_variance", [])))
        pdf.row("Timestamp",      datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        pdf.ln(3)

        pdf.section("2. Algorithm Leaderboard (Silhouette Score)")
        for i, c in enumerate(results.get("clustering", [])):
            sil = c.get("silhouette")
            s   = f"{sil:.4f}" if sil is not None else "Failed"
            n_c = c.get("n_clusters_found", "?")
            pdf.row(_safe(f"#{i+1}  {c['name']}"),
                    f"Silhouette: {s}  |  Clusters found: {n_c}",
                    highlight=(i == 0))
        pdf.ln(3)

        pdf.section("3. Best Hyperparameters (Optuna Tuning)")
        for c in results.get("clustering", []):
            params = {k: v for k, v in c.get("best_params", {}).items()
                      if k not in ("anomalies_found", "outliers_found")}
            if params:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, _safe(c["name"]), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 9)
                for k, v in params.items():
                    pdf.cell(0, 6, _safe(f"    {k}: {v}"), new_x="LMARGIN", new_y="NEXT")
                dbi = c.get("davies_bouldin")
                ch  = c.get("calinski_harabasz")
                if dbi: pdf.cell(0, 6, f"    Davies-Bouldin Index: {dbi:.4f}  (lower=better)", new_x="LMARGIN", new_y="NEXT")
                if ch:  pdf.cell(0, 6, f"    Calinski-Harabasz: {ch:.2f}  (higher=better)", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)

        pdf.section("4. AI Scientific Insights")
        pdf.body(_safe(insight))

        pdf.section("5. System Information")
        pdf.body(
            "Algorithms were selected by the LLM researcher agent (Llama 3.3 70B via Groq). "
            "Hyperparameters were tuned by Optuna TPE sampler. "
            "Silhouette Score is the primary metric (higher = better cluster separation). "
            "PCA was used to reduce dimensionality for visualization."
        )

        pdf.output(report_path)
        return report_path

    # ── PDF report ────────────────────────────────────────────────
    pdf = _PDF()
    pdf.add_page()

    # 1. Overview
    pdf.section("1. Experiment Overview")
    pdf.row("User Prompt",    _safe(user_prompt))
    pdf.row("Task Type",      task.upper())
    pdf.row("Dataset Shape",  str(results.get("dataset_shape", "N/A")))
    pdf.row("Trials / Model", str(results.get("n_trials_per_model", 25)))
    pdf.row("Timestamp",      datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pdf.ln(3)

    # 2. Leaderboard
    pdf.section("2. Model Leaderboard")
    for i, m in enumerate(results.get("models", [])):
        score = m.get("score")
        score_str = f"{score:.4f}" if score is not None else "Failed"
        pdf.row(_safe(f"#{i+1}  {m['name']}"), f"{metric}: {score_str}", highlight=(i == 0))
    pdf.ln(3)

    # 3. Best Hyperparameters
    pdf.section("3. Best Hyperparameters (Optuna Tuning)")
    for m in results.get("models", []):
        if m.get("best_params"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, _safe(m["name"]), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            for k, v in m["best_params"].items():
                pdf.cell(0, 6, _safe(f"    {k}: {v}"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    # 4. AI Insights
    pdf.section("4. AI Scientific Insights")
    pdf.body(_safe(insight))

    # 5. System note
    pdf.section("5. System Information")
    pdf.body(
        "Models were selected by the LLM researcher agent (Llama 3.3 70B via Groq) based on the "
        "user's natural language instruction. Hyperparameter tuning was performed by Optuna using "
        "the TPE (Tree-structured Parzen Estimator) sampler. Results are sorted by best score."
    )

    pdf.output(report_path)
    return report_path