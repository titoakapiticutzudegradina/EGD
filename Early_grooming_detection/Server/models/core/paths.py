from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent

LICENTA_ROOT = MODELS_DIR.parent

TRAIN_PATH = LICENTA_ROOT / "data/processed/train_windows.csv"
TEST_PATH = LICENTA_ROOT / "data/processed/test_windows.csv"
TEST_CUMULATIVE_PATH = LICENTA_ROOT / "data/processed/test_windows_cumulative.csv"

TRAINED_DIR = MODELS_DIR / "trained"
EVALUATED_DIR = MODELS_DIR / "evaluated"
PLOTS_DIR = MODELS_DIR / "plots"
CACHE_DIR = MODELS_DIR / "cache"
