import sys
from pathlib import Path

_models_dir = Path(__file__).resolve().parents[1]
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

import bootstrap  

from utils.logger import get_logger
import time
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from utils.utils import load_data

logger = get_logger("baseline_train", "logs/train.log")


def main():

    start_time = time.time()

    logger.info("Loading data...")
    train_df, _ = load_data("data/processed/train_windows.csv","data/processed/test_windows.csv")


    total = len(train_df)
    positives = train_df["label"].sum()
    negatives = total - positives

    logger.info(f"Dataset size: {total}")
    logger.info(f"Positive samples: {positives}")
    logger.info(f"Negative samples: {negatives}")

    train_split_df, val_split_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["label"]
    )

    logger.info(f"Training split size: {len(train_split_df)}")
    logger.info(f"Validation split size: {len(val_split_df)}")

    #tfidf vectorizer
    logger.info("Vectorizing...")
    vectorizer = TfidfVectorizer(
        max_features=20000,#maximum number of features to consider
        ngram_range=(1, 2),
        min_df=2,#minimum document frequency
        max_df=0.9, #maximum document frequency
        stop_words="english"#common words to ignore
    )

    X_train = vectorizer.fit_transform(train_split_df["text"])
    y_train = train_split_df["label"]
    X_val = vectorizer.transform(val_split_df["text"])
    y_val = val_split_df["label"]

    #logistic regression model
    logger.info("Training model...")
    model = LogisticRegression(
        max_iter=2000, #maximum number of iterations
        class_weight="balanced" #balance the classes
    )

    model.fit(X_train, y_train)

    logger.info("Running validation...")
    val_preds = model.predict(X_val)
    val_f1 = f1_score(y_val, val_preds)
    logger.info(f"Validation F1-score: {val_f1:.4f}")
    for line in str(classification_report(y_val, val_preds)).splitlines():
        logger.info(line)

    os.makedirs("models/trained", exist_ok=True)

    joblib.dump(model, "models/trained/baseline_model.joblib")
    joblib.dump(vectorizer, "models/trained/tfidf_vectorizer.joblib")


    end_time = time.time()
    duration = end_time - start_time

    logger.info(f"Training time: {duration:.2f} seconds")
    logger.info("Model saved successfully.")


if __name__ == "__main__":
    main()
