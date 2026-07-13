try:
    from core.predictors import threshold_for_progress
    from .conversation_text import messages_to_text
except ImportError: 
    from conversation_text import messages_to_text
    from predictors import threshold_for_progress


def resolve_threshold(predictor, *, progress: float | None = None) -> float:
    default = 0.5
    if hasattr(predictor, "threshold"):
        default = float(predictor.threshold)
    elif hasattr(predictor, "THRESHOLD"):
        default = float(predictor.THRESHOLD)

    if progress is None:
        return default

    progress_thresholds = getattr(predictor, "progress_thresholds", None) or {}
    if progress_thresholds:
        return threshold_for_progress(progress, progress_thresholds, default)
    return default


def predictor_threshold(predictor) -> float:
    return resolve_threshold(predictor)


def score_text(
    predictor, text: str, *, progress: float | None = None
) -> float:
    if not text.strip():
        return 0.0
    batch_progress = [progress] if progress is not None else None
    return float(predictor.predict_proba([text], progress=batch_progress)[0])

#get attributable messages
def attributable_messages(
    predictor,
    payload: list[dict],
    *,
    top_k: int = 10,
    progress: float | None = None,
) -> list[dict]:
    threshold = resolve_threshold(predictor, progress=progress)
    full_text = messages_to_text(payload)
    full_prob = score_text(predictor, full_text, progress=progress)

    if full_prob < threshold:
        return []

    non_empty = [
        i for i, msg in enumerate(payload) if (msg.get("text") or "").strip()
    ]
    if not non_empty:
        return []

    if len(non_empty) == 1:
        return [
            _entry(payload, non_empty[0], contribution=full_prob)
        ][:top_k]

    scored: list[dict] = []
    for idx in non_empty:
        reduced = [m for j, m in enumerate(payload) if j != idx]
        reduced_text = messages_to_text(reduced)
        if not reduced_text:
            contribution = full_prob
        else:
            prob_without = score_text(predictor, reduced_text, progress=progress)
            contribution = max(0.0, full_prob - prob_without)
        scored.append(_entry(payload, idx, contribution=contribution))

    scored.sort(key=lambda x: x["contribution"], reverse=True)
    return scored[: max(1, top_k)]

#create entry for a message
def _entry(payload: list[dict], index: int, *, contribution: float) -> dict:
    msg = payload[index]
    return {
        "index": index,
        "text": msg.get("text", ""),
        "contribution": round(contribution, 4),
    }
