"""Train the portfolio version of the DACON log-risk classifier.

The original work was developed in notebooks. This script keeps the final
pipeline in a small, reproducible form:

1. load DACON train/test/submission CSV files,
2. tokenize `full_log` text with Keras Tokenizer,
3. pad each sequence to a fixed length,
4. train an XGBoost multiclass classifier,
5. write a DACON-style submission CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost log-risk classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_xgboost.csv"))
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    return parser.parse_args()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    submission = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, submission


def vectorize_logs(
    train_text: pd.Series,
    val_text: pd.Series,
    test_text: pd.Series,
    max_length: int,
) -> tuple[tf.keras.preprocessing.text.Tokenizer, object, object, object]:
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train_text)

    x_train = tokenizer.texts_to_sequences(train_text)
    x_val = tokenizer.texts_to_sequences(val_text)
    x_test = tokenizer.texts_to_sequences(test_text)

    x_train_vector = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=max_length, padding="post"
    )
    x_val_vector = tf.keras.preprocessing.sequence.pad_sequences(
        x_val, maxlen=max_length, padding="post"
    )
    x_test_vector = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=max_length, padding="post"
    )
    return tokenizer, x_train_vector, x_val_vector, x_test_vector


def train_model(
    x_train,
    y_train,
    x_val,
    y_val,
    n_estimators: int,
    random_state: int,
    early_stopping_rounds: int,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        objective="multi:softprob",
        eval_metric="merror",
    )

    try:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=True,
        )
    except TypeError:
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=True)

    return model


def main() -> None:
    args = parse_args()
    train, test, submission = load_data(args.data_dir)

    text_train, text_val, label_train, label_val = train_test_split(
        train["full_log"],
        train["level"],
        test_size=args.test_size,
        random_state=args.random_state,
    )

    _, x_train_vector, x_val_vector, x_test_vector = vectorize_logs(
        text_train,
        text_val,
        test["full_log"],
        max_length=args.max_length,
    )

    model = train_model(
        x_train_vector,
        label_train,
        x_val_vector,
        label_val,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    val_pred = model.predict(x_val_vector)
    macro_f1 = metrics.f1_score(label_val, val_pred, average="macro")
    print(f"validation macro_f1={macro_f1:.6f}")

    submission["level"] = model.predict(x_test_vector)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
