import sys
from pathlib import Path

_models_dir = Path(__file__).resolve().parents[1]
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

import bootstrap  

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from core.predictors import get_predictor, list_models, results_path, threshold_for_progress
from core.window_sampling import filter_cumulative_windows
from core.training_utils import ROOT
from utils.logger import get_logger
from utils.utils import early_detection_from_preds


TEST_PATH = ROOT / "data/processed/test_windows.csv"
TEST_CUMULATIVE_PATH = ROOT / "data/processed/test_windows_cumulative.csv"
INFERENCE_BATCH = 256
LOG_EVERY_BATCHES = 20


def load_test_dataframe(logger) -> pd.DataFrame:
    if TEST_CUMULATIVE_PATH.is_file():
        logger.info(f"Loading pre-filtered test data from {TEST_CUMULATIVE_PATH}")
        return pd.read_csv(TEST_CUMULATIVE_PATH)

    logger.info(
        f"No {TEST_CUMULATIVE_PATH.name} found; scanning {TEST_PATH} in chunks. "
        "Run build_test_cumulative_csv.py once for faster loads."
    )
    parts: list[pd.DataFrame] = []
    total_in = 0
    for chunk in pd.read_csv(TEST_PATH, chunksize=500_000):
        total_in += len(chunk)
        filtered = filter_cumulative_windows(chunk)
        if not filtered.empty:
            parts.append(filtered)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    logger.info(
        f"Using cumulative windows only ({len(df)} rows, "
        f"dropped {total_in - len(df)} fixed-window rows)"
    )
    return df


def evaluate_full(y, preds, logger):
    report_text = str(classification_report(y, preds, output_dict=False))
    logger.info("Full classification report:")
    for line in report_text.splitlines():
        logger.info(line)

    f1 = f1_score(y, preds)
    logger.info(f"F1-score: {f1:.4f}")


def evaluate_early(df, preds, logger, *, probs=None, progress_thresholds=None, default_threshold=0.5):
    logger.info("Early Detection Evaluation:")
    if probs is not None and progress_thresholds:
        logger.info(f"Using per-progress thresholds: {progress_thresholds}")
        results = early_detection_from_preds(
            df,
            preds,
            probs=probs,
            thresholds_by_progress=progress_thresholds,
            default_threshold=default_threshold,
        )
    else:
        results = early_detection_from_preds(df, preds)

    for cp, f1 in results:
        logger.info(f"{int(cp * 100)}% → F1: {f1:.4f}")

    return results


def save_results(model_name: str, results: list[tuple[float, float]]):
    out = results_path(model_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results, columns=["progress", "f1"]).to_csv(out, index=False)
    return out


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
                    >= threshold_for_progress(float(pr), progress_thresholds, threshold)
                )
                for p, pr in zip(probs, progress)
            ],
            dtype=int,
        )
    return (probs >= threshold).astype(int)


def run_inference(
    predictor,
    df: pd.DataFrame,
    logger,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    progress = (
        df["progress"].astype(float).tolist() if "progress" in df.columns else None
    )
    texts = df["text"].tolist()
    total = len(texts)
    probs = np.empty(total, dtype=np.float64)

    logger.info(f"Running inference on {total} rows (batch_size={batch_size})...")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_progress = progress[start:end] if progress is not None else None
        probs[start:end] = predictor.predict_proba(
            batch_texts, progress=batch_progress
        )
        if end == total or end % (batch_size * LOG_EVERY_BATCHES) == 0:
            logger.info(f"  {end}/{total} rows scored")

    preds = probs_to_labels(predictor, probs, progress)
    return probs, preds


def run(model_name: str, df: pd.DataFrame, *, batch_size: int):
    logger = get_logger(f"eval_{model_name}", f"logs/eval_{model_name}.log")
    logger.info(f"Evaluating model: {model_name}")

    predictor = get_predictor(model_name)
    if hasattr(predictor, "device"):
        logger.info(f"Using device: {predictor.device}")

    probs, preds = run_inference(predictor, df, logger, batch_size=batch_size)

    progress_thresholds = getattr(predictor, "progress_thresholds", None) or {}
    default_thr = float(getattr(predictor, "threshold", 0.5))

    evaluate_full(df["label"].values, preds, logger)
    results = evaluate_early(
        df,
        preds,
        logger,
        probs=probs,
        progress_thresholds=progress_thresholds if progress_thresholds else None,
        default_threshold=default_thr,
    )
    out = save_results(model_name, results)

    logger.info(f"Results saved to {out}")
    logger.info("Evaluation completed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on the test set."
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        choices=list_models(),
        required=True,
        metavar="MODEL",
        help=f"Model(s) to evaluate. Choices: {', '.join(list_models())}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=INFERENCE_BATCH,
        help=f"Rows per inference batch (default: {INFERENCE_BATCH}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger = get_logger("evaluation", "logs/eval.log")
    logger.info("Loading test data...")
    df = load_test_dataframe(logger)

    total = len(df)
    positives = df["label"].sum()
    negatives = total - positives
    logger.info(f"Test dataset size: {total}")
    logger.info(f"Positive samples: {positives}")
    logger.info(f"Negative samples: {negatives}")

    for model_name in args.model:
        run(model_name, df, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
