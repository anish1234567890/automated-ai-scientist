"""
experiment_chat.py  —  Automated AI Scientist v3.0
────────────────────────────────────────────────────
Multi-turn Experiment Chat Agent.

What it does:
  After an experiment completes, the user can ask follow-up questions
  about the results in natural language. The agent:
  1. Maintains conversation history (multi-turn)
  2. Has full context: results, SHAP, health, hypothesis, cluster profiles
  3. Answers questions like:
     "Why did XGBoost beat Random Forest?"
     "What would happen if I removed the smoker column?"
     "Which features should I engineer next?"
     "Is my model overfitting?"
     "Explain the cluster 0 profile in business terms"
  4. Each answer is grounded in the actual experiment data — no hallucination

This is fundamentally different from just showing an insight paragraph.
The user can INTERROGATE the results, not just read a summary.
"""

from typing import List, Dict


def _build_context(
    results:         dict,
    health:          dict,
    hypothesis:      str,
    shap_result:     dict,
    cluster_profile: dict,
    mode:            str,
) -> str:
    """
    Build a rich context string injected into every chat turn.
    The LLM always has the full experiment state.
    """
    lines = [f"=== EXPERIMENT CONTEXT (mode: {mode.upper()}) ==="]

    # Dataset stats
    shape = results.get("dataset_shape", [])
    if shape:
        lines.append(f"Dataset: {shape[0]} rows × {shape[1]} columns")

    # Health
    if health:
        lines.append(
            f"Data Health: Grade {health.get('grade','?')} ({health.get('score',0)}/100)"
        )
        for issue in health.get("issues", [])[:3]:
            lines.append(f"  Health issue: {issue['message'][:100]}")

    if mode == "supervised":
        task   = results.get("task", "unknown")
        metric = "Accuracy" if task == "classification" else "RMSE"
        lines.append(f"Task: {task.upper()} | Metric: {metric} (5-fold CV)")

        # Hypothesis
        if hypothesis:
            lines.append(f"Pre-experiment hypothesis: {hypothesis[:200]}...")

        # Model results
        lines.append("Model leaderboard:")
        for m in results.get("models", [])[:6]:
            score = m.get("score")
            if score is not None:
                params = m.get("best_params", {})
                p_str  = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3])
                lines.append(f"  {m['name']:25s}: {metric}={score:.4f} | params: {p_str}")
            else:
                lines.append(f"  {m['name']:25s}: FAILED — {m.get('error','?')}")

        # Ensemble
        ens = results.get("ensemble", {})
        if ens and not ens.get("error"):
            lines.append(
                f"Ensemble (Top-3 Voting): {metric}={ens.get('cv_score','?')} "
                f"| Models: {', '.join(ens.get('models_used',[]))}"
            )

        # SHAP
        if shap_result and not shap_result.get("error"):
            top5 = shap_result.get("top_features", [])[:5]
            feat_str = ", ".join(
                f"{t['feature']}(SHAP={t['importance']:.4f})" for t in top5
            )
            lines.append(f"SHAP top features: {feat_str}")
            lines.append(f"SHAP explainer used: {shap_result.get('explainer_type','?')}")

    else:  # unsupervised
        lines.append(f"Task: UNSUPERVISED CLUSTERING")
        lines.append("Clustering results:")
        for c in results.get("clustering", [])[:4]:
            sil = c.get("silhouette")
            if sil is not None:
                lines.append(
                    f"  {c['name']:25s}: Silhouette={sil:.4f} | "
                    f"Clusters={c.get('n_clusters_found','?')} | "
                    f"Noise={c.get('n_noise_points',0)}"
                )

        # Cluster profiles
        if cluster_profile and cluster_profile.get("clusters"):
            lines.append("Cluster profiles (best algorithm):")
            for cl in cluster_profile["clusters"]:
                defs    = cl.get("defining_features", [])[:3]
                def_str = ", ".join(
                    f"{d['feature']}={d['cluster_mean']}({d['direction']})"
                    for d in defs
                )
                lines.append(
                    f"  Cluster {cl['label']}: {cl['size']} rows ({cl['pct']}%) — {def_str}"
                )

    lines.append("=== END CONTEXT ===")
    return "\n".join(lines)


def answer_question(
    question:        str,
    history:         List[Dict],
    results:         dict,
    health:          dict,
    hypothesis:      str,
    shap_result:     dict,
    cluster_profile: dict,
    mode:            str,
) -> str:
    """
    Answer a follow-up question about the experiment.
    Multi-turn: history is a list of alternating user/assistant dicts.
    """
    from core.researcher import _call_groq

    # Build compact context (keep under ~800 tokens)
    context = _build_context(
        results or {}, health or {}, hypothesis or "",
        shap_result or {}, cluster_profile or {}, mode or "supervised"
    )

    # ── Correct message structure ─────────────────────────────────
    # Groq (llama) doesn't support "system" role — embed context in
    # the FIRST user message, then alternate user/assistant naturally.

    # First message always = context + the very first question ever asked
    # (or a primer if there's no history yet)
    first_content = (
        f"You are the AI Scientist. You just ran an ML experiment. "
        f"Answer questions strictly using the data below.\n\n"
        f"{context}\n\n"
        f"Rules:\n"
        f"- Be specific: cite exact model names, scores, feature names\n"
        f"- 3-5 sentences unless more detail is needed\n"
        f"- If something isn't in the data, say so honestly\n"
        f"- First person, no bullet points\n\n"
        f"Ready. User will now ask questions."
    )

    messages: List[Dict] = []

    if not history:
        # No history — single turn: context + question in one message
        messages = [{
            "role":    "user",
            "content": first_content + f"\n\nQuestion: {question}"
        }]
    else:
        # Multi-turn: inject context into the very first message,
        # then replay history, then add current question.
        # Take last 6 turns max to stay inside token budget.
        recent = history[-6:]

        # First message = context framing + oldest question in window
        first_q = recent[0]["content"] if recent[0]["role"] == "user" else question
        messages.append({
            "role":    "user",
            "content": first_content + f"\n\nQuestion: {first_q}"
        })

        # Replay the rest of the history (skip the first entry we just used)
        for turn in recent[1:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        # Ensure last message is the current question
        # (only add if it's not already the last entry)
        if not messages or messages[-1].get("content") != question:
            messages.append({"role": "user", "content": question})

    return _call_groq(messages, max_tokens=500)


def get_suggested_questions(results: dict, mode: str, shap_result: dict) -> List[str]:
    """
    Generate 4 context-aware suggested follow-up questions based on what happened.
    Shown in the UI as clickable buttons so users know what they can ask.
    """
    task = results.get("task", "unknown")

    if mode == "supervised":
        valid  = [m for m in results.get("models", []) if m.get("score") is not None]
        best   = valid[0]["name"] if valid else "the best model"
        worst  = valid[-1]["name"] if len(valid) > 1 else "other models"
        metric = "accuracy" if task == "classification" else "RMSE"

        suggestions = [
            f"Why did {best} outperform {worst}?",
            "What do the best hyperparameters tell us about this data?",
        ]

        if shap_result and not shap_result.get("error"):
            top = shap_result.get("top_features", [])
            if top:
                suggestions.append(
                    f"Why is '{top[0]['feature']}' the most important feature?"
                )
        else:
            suggestions.append("Which features are most important for prediction?")

        suggestions.append(f"How can I improve the {metric} further?")
        if task == "classification":
            suggestions.append("Are there signs of overfitting in these results?")
        else:
            suggestions.append("Should I try any feature transformations?")

    else:  # unsupervised
        clustering = [c for c in results.get("clustering", [])
                      if c.get("silhouette") is not None]
        best_algo  = clustering[0]["name"] if clustering else "the algorithm"

        suggestions = [
            f"What do the clusters found by {best_algo} represent in business terms?",
            "Why is the silhouette score relatively low?",
            "How many clusters should I actually use for this data?",
            "Which features drive the most separation between clusters?",
        ]

    return suggestions[:4]
