from __future__ import annotations

import sys
from pathlib import Path

_models_dir = Path(__file__).resolve().parents[1]
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

import bootstrap 

import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from core.paths import EVALUATED_DIR
from core.predictors import get_predictor, list_models, threshold_for_progress
from eval.evaluate import load_test_dataframe
from utils.logger import get_logger

DEFAULT_ERRORS_DIR = EVALUATED_DIR / "errors"
PREVIEW_CHARS = 500
INFERENCE_BATCH = 256

TAG_PATTERNS: dict[str, re.Pattern[str]] = {
    "casual_opener": re.compile(
        r"\b(asl|where are you from|how old|w r u|stranger|hous it going)\b",
        re.I,
    ),
    "explicit": re.compile(
        r"\b(webcam|masturbat|nude|naked|horny|sexy|cum on|pic\?|pics\?)\b",
        re.I,
    ),
    "technical": re.compile(
        r"\b(w3\.org|bugzilla|firefox|validator|html|webkit|xml)\b",
        re.I,
    ),
    "emotional": re.compile(
        r"\b(love|miss you|sweet dreams|dont be mad|feel so bad)\b",
        re.I,
    ),
}


def hint_tags(text: str, *, text_length: int) -> str:
    tags: list[str] = []
    for name, pattern in TAG_PATTERNS.items():
        if pattern.search(text):
            tags.append(name)
    if text_length < 200:
        tags.append("short")
    return ",".join(tags)


def message_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def select_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "windows":
        return df.reset_index(drop=True)
    if scope == "conversation":
        if "conv_id" not in df.columns:
            raise ValueError("conv_id column required for conversation scope.")
        idx = df.groupby("conv_id")["progress"].idxmax()
        return df.loc[idx].reset_index(drop=True)
    raise ValueError(f"Unknown scope {scope!r}. Use 'conversation' or 'windows'.")


def run_inference(
    predictor,
    df: pd.DataFrame,
    *,
    batch_size: int,
    logger,
) -> tuple[np.ndarray, np.ndarray]:
    progress = (
        df["progress"].astype(float).tolist() if "progress" in df.columns else None
    )
    texts = df["text"].tolist()
    probs_parts: list[np.ndarray] = []
    total = len(texts)

    logger.info(f"Running inference on {total} rows (batch_size={batch_size})...")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_progress = progress[start:end] if progress is not None else None
        batch_probs = predictor.predict_proba(batch_texts, progress=batch_progress)
        probs_parts.append(np.asarray(batch_probs, dtype=float))
        if end == total or end % (batch_size * 20) == 0:
            logger.info(f"  {end}/{total} rows scored")

    probs = np.concatenate(probs_parts) if probs_parts else np.array([], dtype=float)
    preds = probs_to_labels(predictor, probs, progress)
    return probs, preds


def probs_to_labels(
    predictor,
    probs: np.ndarray,
    progress: list[float] | None,
) -> np.ndarray:
    threshold = float(getattr(predictor, "threshold", getattr(predictor, "THRESHOLD", 0.5)))
    progress_thresholds = getattr(predictor, "progress_thresholds", None) or {}
    if progress is not None and progress_thresholds:
        return np.array(
            [
                int(
                    p
                    >= threshold_for_progress(
                        float(pr), progress_thresholds, threshold
                    )
                )
                for p, pr in zip(probs, progress)
            ],
            dtype=int,
        )
    return (probs >= threshold).astype(int)


def build_results_frame(
    df: pd.DataFrame,
    probs: np.ndarray,
    preds: np.ndarray,
) -> pd.DataFrame:
    out = df.copy()
    out["score"] = np.round(probs, 4)
    out["pred"] = preds
    out["correct"] = (out["label"] == out["pred"]).astype(int)

    conditions = [
        (out["label"] == 0) & (out["pred"] == 1),
        (out["label"] == 1) & (out["pred"] == 0),
        (out["label"] == 1) & (out["pred"] == 1),
        (out["label"] == 0) & (out["pred"] == 0),
    ]
    choices = ["FP", "FN", "TP", "TN"]
    out["error_type"] = np.select(conditions, choices, default="?")

    lengths = out["text"].str.len()
    out["text_length"] = lengths
    out["message_count"] = out["text"].map(message_count)
    out["tags"] = [
        hint_tags(text, text_length=int(length))
        for text, length in zip(out["text"], lengths)
    ]
    out["text_preview"] = out["text"].str.slice(0, PREVIEW_CHARS)
    return out


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "f1_predatory": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": report,
    }


def export_errors(
    results: pd.DataFrame,
    out_dir: Path,
    *,
    max_samples: int | None,
    save_all: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    export_cols = [
        "conv_id",
        "progress",
        "label",
        "pred",
        "score",
        "error_type",
        "text_length",
        "message_count",
        "tags",
        "text_preview",
        "text",
    ]
    export_cols = [c for c in export_cols if c in results.columns]

    fp = results[results["error_type"] == "FP"].sort_values("score", ascending=False)
    fn = results[results["error_type"] == "FN"].sort_values("score", ascending=True)

    if max_samples is not None:
        fp = fp.head(max_samples)
        fn = fn.head(max_samples)

    errors = pd.concat([fp, fn], ignore_index=True)

    fp[export_cols].to_csv(out_dir / "false_positives.csv", index=False)
    fn[export_cols].to_csv(out_dir / "false_negatives.csv", index=False)
    errors[export_cols].to_csv(out_dir / "errors.csv", index=False)

    if save_all:
        results[export_cols].to_csv(out_dir / "all_predictions.csv", index=False)


def tag_summary(errors: pd.DataFrame) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {"FP": {}, "FN": {}}
    for error_type in ("FP", "FN"):
        subset = errors[errors["error_type"] == error_type]
        counts: dict[str, int] = {}
        for tags in subset["tags"]:
            for tag in (tags or "").split(","):
                tag = tag.strip()
                if not tag:
                    continue
                counts[tag] = counts.get(tag, 0) + 1
        summary[error_type] = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
    return summary


def analyze_model(
    model_name: str,
    df: pd.DataFrame,
    *,
    out_root: Path,
    max_samples: int | None,
    save_all: bool,
    batch_size: int,
    logger,
) -> dict:
    logger.info(f"Analyzing model: {model_name}")
    predictor = get_predictor(model_name)
    if hasattr(predictor, "device"):
        logger.info(f"Using device: {predictor.device}")

    probs, preds = run_inference(predictor, df, batch_size=batch_size, logger=logger)
    results = build_results_frame(df, probs, preds)

    y_true = results["label"].values
    y_pred = results["pred"].values
    summary = summarize(y_true, y_pred)
    summary["model"] = model_name
    summary["rows"] = int(len(results))
    summary["positives"] = int(results["label"].sum())
    summary["negatives"] = int(len(results) - results["label"].sum())

    errors_only = results[results["error_type"].isin(["FP", "FN"])]
    summary["tag_counts"] = tag_summary(errors_only)

    model_dir = out_root / model_name
    export_errors(results, model_dir, max_samples=max_samples, save_all=save_all)

    summary_path = model_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    cm = summary["confusion_matrix"]
    logger.info(
        f"{model_name}: F1={summary['f1_predatory']:.4f} "
        f"FP={cm['fp']} FN={cm['fn']} TP={cm['tp']} TN={cm['tn']}"
    )
    logger.info(f"Wrote errors to {model_dir}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run error analysis on the test set: export false positives, "
            "false negatives, and per-model summaries."
        )
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        choices=list_models(),
        required=True,
        metavar="MODEL",
        help=f"Model(s) to analyze. Choices: {', '.join(list_models())}",
    )
    parser.add_argument(
        "--scope",
        choices=("conversation", "windows"),
        default="conversation",
        help=(
            "conversation: one row per conv_id at max progress (default); "
            "windows: all cumulative test rows (matches evaluate.py)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_ERRORS_DIR,
        help=f"Root directory for error exports (default: {DEFAULT_ERRORS_DIR})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max FP and FN rows to export per model (default: 200). Use 0 for all.",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Also write all_predictions.csv for each model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=INFERENCE_BATCH,
        help=f"Inference batch size (default: {INFERENCE_BATCH}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    max_samples = None if args.max_samples == 0 else args.max_samples

    logger = get_logger("error_analysis", "logs/error_analysis.log")
    logger.info("Loading test data...")
    df = load_test_dataframe(logger)
    if df.empty:
        raise SystemExit("Test dataset is empty.")

    df = select_scope(df, args.scope)
    logger.info(f"Scope={args.scope}: {len(df)} rows, {int(df['label'].sum())} positives")

    summaries: list[dict] = []
    for model_name in args.model:
        summaries.append(
            analyze_model(
                model_name,
                df,
                out_root=args.output_dir,
                max_samples=max_samples,
                save_all=args.save_all,
                batch_size=args.batch_size,
                logger=logger,
            )
        )

    if len(summaries) > 1:
        combined_path = args.output_dir / "comparison.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        combined_path.write_text(json.dumps(summaries, indent=2) + "\n")
        logger.info(f"Wrote comparison to {combined_path}")


if __name__ == "__main__":
    main()
