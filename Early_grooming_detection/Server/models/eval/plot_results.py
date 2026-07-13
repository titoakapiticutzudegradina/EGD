import sys
from pathlib import Path

_models_dir = Path(__file__).resolve().parents[1]
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

import bootstrap  

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from core.paths import EVALUATED_DIR, PLOTS_DIR
from core.predictors import RESULT_FILES, list_models


def output_filename(models: list[str]) -> str:
    if len(models) == 1:
        return f"{models[0]}_early_detection_curve.png"
    return f"{'_'.join(models)}_early_detection_curve.png"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot early-detection F1 curves.")
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        choices=list_models(),
        default=["baseline"],
        metavar="MODEL",
        help=f"Model(s) to plot. Choices: {', '.join(list_models())}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output filename inside models/plots/ (default: derived from model names)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for model_name in args.model:
        csv_path = EVALUATED_DIR / RESULT_FILES[model_name]
        df = pd.read_csv(csv_path)
        plt.plot(df["progress"], df["f1"], marker="o", label=model_name)

    plt.xlabel("Conversation Progress")
    plt.ylabel("F1 Score")
    title = "Early Detection Performance"
    if len(args.model) > 1:
        title += f" ({', '.join(args.model)})"
    plt.title(title)
    plt.legend()
    plt.grid()
    filename = args.output or output_filename(args.model)
    out_path = PLOTS_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")



if __name__ == "__main__":
    main()
