"""
data_health.py
──────────────
Automatic dataset health report for the Automated AI Scientist.

Checks performed:
  1. Missing values — count + % per column
  2. Class imbalance — only for classification target
  3. Duplicate rows
  4. Constant / near-constant columns (< 2 unique values)
  5. High cardinality string columns (> 50 unique values)
  6. Numeric outliers — IQR method, flags columns with > 5% outliers
  7. Feature correlation — flags pairs > 0.95 (potential multicollinearity)
  8. Dataset size warnings — very small (< 100 rows) or very wide (> 100 cols)

Returns a structured dict with:
  "score"    : int 0-100 (100 = perfect health)
  "grade"    : str "A" / "B" / "C" / "D"
  "issues"   : list of issue dicts {type, severity, message, column}
  "summary"  : one-line plain English summary
  "stats"    : basic dataset statistics
"""

import pandas as pd
import numpy as np
from typing import Optional


def _grade(score: int) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 55: return "C"
    return "D"


def run_health_check(df: pd.DataFrame, task: Optional[str] = None) -> dict:
    """
    Run all health checks on a DataFrame.

    Parameters
    ----------
    df   : raw DataFrame (before any preprocessing)
    task : "classification" / "regression" / None

    Returns
    -------
    health dict
    """
    issues  = []
    penalty = 0   # deducted from 100

    n_rows, n_cols = df.shape
    has_target     = "target" in df.columns
    feature_cols   = [c for c in df.columns if c != "target"]

    # ── 1. Missing values ─────────────────────────────────────────
    null_counts = df[feature_cols].isnull().sum()
    total_cells = n_rows * len(feature_cols)
    total_null  = int(null_counts.sum())
    null_pct    = round(total_null / max(total_cells, 1) * 100, 1)

    if null_pct > 0:
        sev = "high" if null_pct > 20 else ("medium" if null_pct > 5 else "low")
        penalty += 20 if null_pct > 20 else (10 if null_pct > 5 else 3)
        worst_col = null_counts.idxmax() if total_null > 0 else None
        issues.append({
            "type":     "missing_values",
            "severity": sev,
            "message":  f"{total_null} missing values ({null_pct}% of feature cells). "
                        f"Worst column: '{worst_col}' ({null_counts.max()} nulls).",
            "column":   worst_col,
        })

    # ── 2. Class imbalance (classification only) ──────────────────
    if has_target and (task == "classification" or
                       (task is None and df["target"].nunique() < 15)):
        vc    = df["target"].value_counts(normalize=True)
        ratio = vc.min() / vc.max() if vc.max() > 0 else 1.0
        if ratio < 0.1:
            penalty += 25
            issues.append({
                "type":     "class_imbalance",
                "severity": "high",
                "message":  f"Severe class imbalance — minority class is only "
                            f"{round(vc.min()*100, 1)}% of data. "
                            f"Consider SMOTE or class_weight='balanced'.",
                "column":   "target",
            })
        elif ratio < 0.3:
            penalty += 10
            issues.append({
                "type":     "class_imbalance",
                "severity": "medium",
                "message":  f"Moderate class imbalance — minority/majority ratio = "
                            f"{round(ratio, 2)}. Consider adjusting class weights.",
                "column":   "target",
            })

    # ── 3. Duplicate rows ─────────────────────────────────────────
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        dupe_pct = round(n_dupes / n_rows * 100, 1)
        sev = "high" if dupe_pct > 10 else "medium"
        penalty += 15 if dupe_pct > 10 else 5
        issues.append({
            "type":     "duplicate_rows",
            "severity": sev,
            "message":  f"{n_dupes} duplicate rows ({dupe_pct}%). "
                        f"These can inflate model performance metrics.",
            "column":   None,
        })

    # ── 4. Constant / near-constant columns ───────────────────────
    const_cols = [c for c in feature_cols if df[c].nunique() <= 1]
    if const_cols:
        penalty += 5 * len(const_cols)
        issues.append({
            "type":     "constant_columns",
            "severity": "high",
            "message":  f"{len(const_cols)} constant column(s) — carry zero information: "
                        f"{', '.join(const_cols)}. Should be dropped.",
            "column":   const_cols[0],
        })

    # ── 5. High cardinality string columns ────────────────────────
    hc_cols = [c for c in feature_cols
               if df[c].dtype == "object" and df[c].nunique() > 50]
    if hc_cols:
        penalty += 8
        issues.append({
            "type":     "high_cardinality",
            "severity": "medium",
            "message":  f"{len(hc_cols)} high-cardinality string column(s) "
                        f"(> 50 unique values): {', '.join(hc_cols)}. "
                        f"Consider target encoding or dropping.",
            "column":   hc_cols[0],
        })

    # ── 6. Outliers (IQR method) ──────────────────────────────────
    num_cols    = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    outlier_cols = []
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr     = q3 - q1
        if iqr == 0:
            continue
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if n_out / n_rows > 0.05:
            outlier_cols.append((col, n_out))

    if outlier_cols:
        penalty += min(15, 3 * len(outlier_cols))
        worst = max(outlier_cols, key=lambda x: x[1])
        issues.append({
            "type":     "outliers",
            "severity": "medium",
            "message":  f"{len(outlier_cols)} column(s) have > 5% outliers (IQR method). "
                        f"Worst: '{worst[0]}' ({worst[1]} outliers). "
                        f"Consider RobustScaler or winsorization.",
            "column":   worst[0],
        })

    # ── 7. High feature correlation ───────────────────────────────
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().abs()
        upper       = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_pairs  = [
            (c, r)
            for c in upper.columns
            for r in upper.index
            if pd.notna(upper.loc[r, c]) and upper.loc[r, c] > 0.95
        ]
        if high_pairs:
            penalty += min(10, 3 * len(high_pairs))
            p = high_pairs[0]
            issues.append({
                "type":     "high_correlation",
                "severity": "low",
                "message":  f"{len(high_pairs)} highly correlated feature pair(s) (r > 0.95). "
                            f"Example: '{p[0]}' ↔ '{p[1]}'. "
                            f"Consider dropping one from each pair.",
                "column":   p[0],
            })

    # ── 8. Dataset size ───────────────────────────────────────────
    if n_rows < 100:
        penalty += 20
        issues.append({
            "type":     "small_dataset",
            "severity": "high",
            "message":  f"Only {n_rows} rows — very small dataset. "
                        f"Cross-validation results will have high variance. "
                        f"Consider collecting more data.",
            "column":   None,
        })
    elif n_rows < 500:
        penalty += 5
        issues.append({
            "type":     "small_dataset",
            "severity": "low",
            "message":  f"{n_rows} rows — relatively small. "
                        f"5-fold CV helps, but results may be noisy.",
            "column":   None,
        })

    if n_cols > 100:
        penalty += 10
        issues.append({
            "type":     "high_dimensionality",
            "severity": "medium",
            "message":  f"{n_cols} columns — high dimensionality. "
                        f"Consider PCA or aggressive feature selection.",
            "column":   None,
        })

    # ── Compute score + grade ─────────────────────────────────────
    score = max(0, 100 - penalty)
    grade = _grade(score)

    # ── Summary line ──────────────────────────────────────────────
    n_issues = len(issues)
    high     = sum(1 for i in issues if i["severity"] == "high")
    if n_issues == 0:
        summary = "✅ Dataset looks clean — no issues found."
    elif high > 0:
        summary = (f"⚠️ {n_issues} issue(s) found ({high} high-severity). "
                   f"Address high-severity issues before trusting results.")
    else:
        summary = (f"ℹ️ {n_issues} minor issue(s) found. "
                   f"Dataset is usable but consider the recommendations below.")

    # ── Basic stats ───────────────────────────────────────────────
    num_features = len([c for c in feature_cols
                        if df[c].dtype != "object"])
    cat_features = len([c for c in feature_cols
                        if df[c].dtype == "object"])

    stats = {
        "rows":                n_rows,
        "columns":             n_cols,
        "numeric_features":    num_features,
        "categorical_features":cat_features,
        "missing_cells":       total_null,
        "missing_pct":         null_pct,
        "duplicate_rows":      n_dupes,
        "memory_mb":           round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }

    if has_target:
        stats["target_unique_values"] = int(df["target"].nunique())
        stats["target_dtype"]         = str(df["target"].dtype)

    return {
        "score":   score,
        "grade":   grade,
        "issues":  issues,
        "summary": summary,
        "stats":   stats,
    }
