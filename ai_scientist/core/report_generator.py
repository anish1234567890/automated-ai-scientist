import os

try:
    from fpdf import FPDF
    _FPDF_AVAILABLE = True
except ImportError:
    FPDF = object          # dummy base — class definition will not crash
    _FPDF_AVAILABLE = False


def _get_report_path():
    try:
        from config import REPORT_PATH
        return REPORT_PATH
    except Exception:
        return os.path.join("outputs", "report.pdf")


def _safe_text(value, default="N/A"):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_pdf_text(value, default="N/A"):
    """Convert to Latin-1 safe text for core FPDF fonts in fpdf2."""
    text = _safe_text(value, default=default)
    return text.encode("latin-1", "replace").decode("latin-1")


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _iter_dicts(value):
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _safe_score(value, digits=4):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def _safe_delta(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):+.4f}"
    except Exception:
        return "N/A"


def _safe_minutes(seconds):
    if seconds is None:
        return "N/A"
    try:
        return f"{float(seconds) / 60.0:.1f} minutes"
    except Exception:
        return "N/A"


def _normalize_shap_name(name):
    if not name:
        return "N/A"
    name = str(name).strip()
    if name == "tree":
        return "TreeExplainer"
    if name == "linear":
        return "LinearExplainer"
    if name == "kernel":
        return "KernelExplainer"
    return name


def _derive_best_supervised(results):
    results = _as_dict(results)
    task = results.get("task", "classification")
    valid = [m for m in _iter_dicts(results.get("models", [])) if m.get("score") is not None]
    if not valid:
        return None, None
    if task == "classification":
        best = max(valid, key=lambda x: x["score"])
    else:
        best = min(valid, key=lambda x: x["score"])
    return best.get("name"), best.get("score")


def _derive_best_unsupervised(results):
    results = _as_dict(results)
    valid = [c for c in _iter_dicts(results.get("clustering", [])) if c.get("silhouette") is not None]
    if not valid:
        return None, None
    best = max(valid, key=lambda x: x["silhouette"])
    return best.get("name"), best.get("silhouette")


class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Automated AI Scientist Report", ln=True, align="C")
        self.ln(2)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, title, ln=True)
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, _safe_pdf_text(text, default="N/A"))
        self.ln(1)

    def kv_row(self, label, value, label_w=60):
        self.set_font("Helvetica", "B", 10)
        self.cell(label_w, 7, _safe_pdf_text(label, default=""), border=1)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 7, _safe_pdf_text(value), border=1, ln=True)

    def table_header(self, headers, widths):
        self.set_font("Helvetica", "B", 9)
        for h, w in zip(headers, widths):
            self.cell(w, 7, _safe_pdf_text(h, default=""), border=1, align="C")
        self.ln()

    def table_row(self, values, widths):
        self.set_font("Helvetica", "", 9)
        row_height = 7
        start_x = self.get_x()
        start_y = self.get_y()

        max_lines = 1
        normalized = []
        for value, width in zip(values, widths):
            text = _safe_pdf_text(value)
            normalized.append(text)
            lines = max(1, len(text) // max(1, int(width / 2.5)) + 1)
            max_lines = max(max_lines, lines)

        row_h = row_height * max_lines
        x = start_x
        y = start_y

        for text, width in zip(normalized, widths):
            self.rect(x, y, width, row_h)
            self.set_xy(x + 1, y + 1)
            self.multi_cell(width - 2, row_height - 1, text, border=0)
            x += width

        self.set_xy(start_x, start_y + row_h)

    def spacer(self, h=2):
        self.ln(h)


def generate_pdf_report(
    results,
    insight,
    user_prompt,
    mode="supervised",
    round1_score=None,
    round1_model=None,
    round2_score=None,
    round2_model=None,
    delta=None,
    shap_explainer=None,
    ensemble_score=None,
    health_grade=None,
    experiment_time_sec=None,
    dataset_name=None,
    features_added=None,
):
    report_path = _safe_text(_get_report_path(), default=os.path.join("outputs", "report.pdf"))
    if report_path == "N/A":
        report_path = os.path.join("outputs", "report.pdf")

    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    results = _as_dict(results)

    task = results.get("task", "unknown")
    dataset_shape = results.get("dataset_shape", [])

    if mode == "supervised":
        derived_model, derived_score = _derive_best_supervised(results)
    else:
        derived_model, derived_score = _derive_best_unsupervised(results)

    if round1_model is None:
        round1_model = derived_model
    if round1_score is None:
        round1_score = derived_score

    if delta is None and round1_score is not None and round2_score is not None:
        try:
            delta = float(round2_score) - float(round1_score)
        except Exception:
            delta = None

    shap_info = _as_dict(results.get("shap"))
    ensemble_info = _as_dict(results.get("ensemble"))

    if shap_explainer is None:
        shap_explainer = _normalize_shap_name(shap_info.get("explainer_type"))
    else:
        shap_explainer = _normalize_shap_name(shap_explainer)

    if ensemble_score is None:
        ensemble_score = ensemble_info.get("cv_score")

    if features_added is None:
        features_added = 0

    if not _FPDF_AVAILABLE:
        txt_path = report_path.replace(".pdf", ".txt")
        _dir = os.path.dirname(txt_path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Automated AI Scientist Report\n")
            f.write(f"Mode: {mode} | Task: {task}\n")
            f.write(f"Dataset Shape: {dataset_shape}\n")
            f.write(f"Insight: {insight}\n")
            f.write(f"Round 1 Best: {round1_model} ({_safe_score(round1_score)})\n")
            f.write(f"Round 2 Best: {round2_model} ({_safe_score(round2_score)})\n")
            f.write(f"Delta: {_safe_delta(delta)}\n")
            f.write(f"SHAP: {shap_explainer} | Ensemble: {_safe_score(ensemble_score)}\n")
            f.write(f"Health Grade: {health_grade} | Features Added: {features_added}\n")
            f.write(f"Run Time: {_safe_minutes(experiment_time_sec)}\n")
        print(f"Saved text report to {txt_path} (fpdf2 not installed)")
        return txt_path

    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 1. Experiment Overview
    pdf.section_title("1. Experiment Overview")
    pdf.kv_row("Mode", mode)
    pdf.kv_row("Task", task)
    pdf.kv_row("Dataset Shape", str(dataset_shape))
    pdf.kv_row("User Prompt", user_prompt)
    pdf.spacer()

    # 2. Results
    pdf.section_title("2. Results")
    if mode == "supervised":
        headers = ["Model", "Score", "Trials", "Best Params"]
        widths = [42, 24, 22, 102]
        pdf.table_header(headers, widths)

        for m in _iter_dicts(results.get("models", [])):
            score = _safe_score(m.get("score"))
            trials = _safe_text(m.get("n_trials"), default="-")
            params = _safe_text(m.get("best_params"), default="{}")
            pdf.table_row([m.get("name", "N/A"), score, trials, params], widths)

        ens = _as_dict(results.get("ensemble"))
        if ens and not ens.get("error"):
            models_used = ens.get("models_used") if isinstance(ens.get("models_used"), list) else []
            pdf.spacer()
            pdf.kv_row("Ensemble Models", ", ".join(str(x) for x in models_used))
            pdf.kv_row("Ensemble CV Score", _safe_score(ens.get("cv_score")))
    else:
        headers = ["Algorithm", "Silhouette", "DB Index", "CH Score", "Best Params"]
        widths = [38, 24, 24, 28, 76]
        pdf.table_header(headers, widths)

        for c in _iter_dicts(results.get("clustering", [])):
            pdf.table_row([
                c.get("name", "N/A"),
                _safe_score(c.get("silhouette")),
                _safe_score(c.get("davies_bouldin")),
                _safe_score(c.get("calinski_harabasz")),
                _safe_text(c.get("best_params"), default="{}"),
            ], widths)

    pdf.spacer()

    # 3. AI Insight
    pdf.section_title("3. AI Insight")
    pdf.body_text(insight)

    # 4. Best Parameters / Diagnostics
    pdf.section_title("4. Best Parameters")
    if mode == "supervised":
        for m in _iter_dicts(results.get("models", [])):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, _safe_pdf_text(m.get("name")), ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, _safe_pdf_text(f"Best Params: {_safe_text(m.get('best_params'), '{}')}", default="Best Params: {}"))
            if m.get("error"):
                pdf.multi_cell(0, 5, _safe_pdf_text(f"Error: {_safe_text(m.get('error'))}"))
            pdf.spacer(1)
    else:
        for c in _iter_dicts(results.get("clustering", [])):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, _safe_pdf_text(c.get("name")), ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, _safe_pdf_text(f"Best Params: {_safe_text(c.get('best_params'), '{}')}", default="Best Params: {}"))
            if c.get("error"):
                pdf.multi_cell(0, 5, _safe_pdf_text(f"Error: {_safe_text(c.get('error'))}"))
            pdf.spacer(1)

    # 5. Final Code
    pdf.section_title("5. Final Code")
    final_code = results.get("final_code", "N/A")
    pdf.set_font("Courier", "", 7)
    pdf.multi_cell(0, 4, _safe_pdf_text(final_code))
    pdf.spacer()

    # 6. Paper Values Summary
    pdf.section_title("6. Paper Values Summary")
    pdf.kv_row("Dataset Name", dataset_name)
    pdf.kv_row("Task Type", task)
    pdf.kv_row("Round 1 Best Model", round1_model)
    pdf.kv_row("Round 1 Best Score", _safe_score(round1_score))
    pdf.kv_row("Round 2 Best Model", round2_model)
    pdf.kv_row("Round 2 Best Score", _safe_score(round2_score))
    pdf.kv_row("Improvement Delta", _safe_delta(delta))
    pdf.kv_row("SHAP Explainer Used", shap_explainer)
    pdf.kv_row("Ensemble Score", _safe_score(ensemble_score))
    pdf.kv_row("Health Grade", health_grade)
    pdf.kv_row("Features Added", str(features_added) if features_added is not None else "N/A")
    pdf.kv_row("Total Run Time", _safe_minutes(experiment_time_sec))

    try:
        pdf.output(report_path)
        print(f"Saved PDF report to {report_path}")
        return report_path
    except Exception as e:
        fallback_path = os.path.join("outputs", "report_fallback.pdf")
        try:
            os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
            pdf.output(fallback_path)
            print(f"Saved PDF report to fallback path {fallback_path} (original error: {e})")
            return fallback_path
        except Exception as e2:
            print(f"Failed to save PDF report (primary and fallback failed): {e2}")
            return report_path