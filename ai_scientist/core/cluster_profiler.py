"""
cluster_profiler.py
───────────────────
Cluster profiling for the Automated AI Scientist.

What it does:
  After clustering finds groups, this module characterises WHAT each
  cluster actually represents by computing per-feature statistics
  per cluster and flagging the most distinctive features.

Output for each cluster:
  - Size (n rows + % of dataset)
  - Mean of every numeric feature
  - Mode of every categorical feature
  - Top 3 "defining features" — features where this cluster deviates
    most from the global mean (z-score based)

This is what turns "Cluster 0 found" into:
  "Cluster 0 (287 rows, 21%): high age (avg 52),
   smoker=yes (100%), high charges (avg $35k)"
"""

import pandas as pd
import numpy as np
from typing import Optional


def profile_clusters(
    df_original: pd.DataFrame,
    labels: list,
    feature_names: list,
    algo_name: str = "",
) -> dict:
    """
    Build a human-readable profile for each cluster.

    Parameters
    ----------
    df_original  : original DataFrame (before scaling) with original column names
    labels       : list of cluster label integers (same length as df)
    feature_names: list of feature column names used during clustering
    algo_name    : name of the clustering algorithm (for labelling)

    Returns
    -------
    {
      "algo": str,
      "n_clusters": int,
      "n_noise": int,
      "clusters": [
        {
          "label": int,
          "size": int,
          "pct": float,
          "numeric_means": {col: mean, ...},
          "categorical_modes": {col: mode, ...},
          "defining_features": [{feature, cluster_mean, global_mean, z_score}, ...]
        }, ...
      ],
      "feature_summary": {col: {cluster_0: mean, cluster_1: mean, ...}, ...}
    }
    """
    labels_arr = np.array(labels)
    n_total    = len(labels_arr)
    if n_total == 0:
        return []

    # Work only with feature columns that exist in the original df
    available  = [c for c in feature_names if c in df_original.columns]
    df_feat    = df_original[available].copy()

    # Attach labels
    df_feat["_cluster"] = labels_arr

    unique_labels = sorted([l for l in np.unique(labels_arr) if l != -1])
    n_noise       = int(np.sum(labels_arr == -1))
    n_clusters    = len(unique_labels)

    # Global stats for z-score computation
    num_cols  = df_feat[available].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols  = df_feat[available].select_dtypes(include=["object"]).columns.tolist()

    global_means = df_feat[num_cols].mean() if num_cols else pd.Series(dtype=float)
    global_stds  = df_feat[num_cols].std().replace(0, 1) if num_cols else pd.Series(dtype=float)

    clusters = []
    for lbl in unique_labels:
        mask      = df_feat["_cluster"] == lbl
        cluster_df = df_feat[mask]
        size      = int(mask.sum())
        pct       = round(size / n_total * 100, 1)

        # Numeric means
        num_means = {}
        if num_cols:
            for col in num_cols:
                val = cluster_df[col].mean()
                num_means[col] = round(float(val), 3) if pd.notna(val) else None

        # Categorical modes
        cat_modes = {}
        for col in cat_cols:
            mode_series = cluster_df[col].mode()
            if len(mode_series) > 0:
                cat_modes[col] = str(mode_series.iloc[0])

        # Defining features — top 3 by z-score vs global mean
        defining = []
        for col in num_cols:
            if col not in num_means or num_means[col] is None:
                continue
            gm  = float(global_means.get(col, 0))
            gs  = float(global_stds.get(col, 1))
            z   = (num_means[col] - gm) / gs if gs != 0 else 0
            defining.append({
                "feature":      col,
                "cluster_mean": num_means[col],
                "global_mean":  round(gm, 3),
                "z_score":      round(z, 3),
                "direction":    "↑ above avg" if z > 0 else "↓ below avg",
            })

        defining.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        defining = defining[:5]   # top 5 defining features

        clusters.append({
            "label":             int(lbl),
            "size":              size,
            "pct":               pct,
            "numeric_means":     num_means,
            "categorical_modes": cat_modes,
            "defining_features": defining,
        })

    # Feature summary table: {feature: {cluster_0: mean, cluster_1: mean, ...}}
    feature_summary = {}
    for col in num_cols:
        row = {"global_mean": round(float(global_means.get(col, 0)), 3)}
        for lbl in unique_labels:
            mask = df_feat["_cluster"] == lbl
            val  = df_feat.loc[mask, col].mean()
            row[f"cluster_{lbl}"] = round(float(val), 3) if pd.notna(val) else None
        feature_summary[col] = row

    return {
        "algo":            algo_name,
        "n_clusters":      n_clusters,
        "n_noise":         n_noise,
        "clusters":        clusters,
        "feature_summary": feature_summary,
    }


def generate_cluster_narrative(profile: dict) -> str:
    """
    Build a plain-English description of each cluster from its profile.
    Used for display in the UI and for injecting into the LLM insight prompt.
    """
    lines = []
    for c in profile.get("clusters", []):
        lbl      = c["label"]
        size     = c["size"]
        pct      = c["pct"]
        defining = c.get("defining_features", [])
        cat_modes = c.get("categorical_modes", {})

        desc_parts = []
        for d in defining[:3]:
            desc_parts.append(
                f"{d['feature']}={d['cluster_mean']} ({d['direction']})"
            )
        for col, val in list(cat_modes.items())[:2]:
            desc_parts.append(f"{col}={val}")

        desc = ", ".join(desc_parts) if desc_parts else "no distinctive features"
        lines.append(f"Cluster {lbl} ({size} rows, {pct}%): {desc}")

    return "\n".join(lines)
