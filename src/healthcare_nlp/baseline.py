"""TF-IDF + Logistic Regression baseline used as a floor for BERT fine-tuning."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

_COL_TEXT = "_norm_text"
_COL_LABEL = "label"
_REQUIRED_COLS = {_COL_TEXT, _COL_LABEL}

logger = logging.getLogger(__name__)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_preprocessed_splits(
    path_to_splits: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(path_to_splits / "train.parquet")
    val = pd.read_parquet(path_to_splits / "val.parquet")
    test = pd.read_parquet(path_to_splits / "test.parquet")

    _validate_columns(train)
    _validate_columns(val)
    _validate_columns(test)

    return train, val, test


@dataclass(frozen=True)
class SplitMetrics:
    accuracy: float
    macro_f1: float
    report: str


def evaluate_split(y_true: pd.Series, y_pred: pd.Series) -> SplitMetrics:
    return SplitMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        report=cast(str, classification_report(y_true, y_pred, digits=4)),
    )


def run_tfidf_logreg_baseline(
    path_to_splits: Path,
    *,
    max_features: int = 20_000,
    ngram_range: tuple[int, int] = (1, 2),
    c: float = 1.0,
) -> tuple[TfidfVectorizer, LogisticRegression, SplitMetrics, SplitMetrics]:
    """Fit TF-IDF + class-balanced Logistic Regression and return fitted objects plus metrics."""
    train_df, val_df, test_df = load_preprocessed_splits(path_to_splits)

    x_train = cast(pd.Series, train_df[_COL_TEXT])
    y_train = cast(pd.Series, train_df[_COL_LABEL])
    x_val = cast(pd.Series, val_df[_COL_TEXT])
    y_val = cast(pd.Series, val_df[_COL_LABEL])
    x_test = cast(pd.Series, test_df[_COL_TEXT])
    y_test = cast(pd.Series, test_df[_COL_LABEL])

    logger.info(
        "Dataset sizes | train=%d val=%d test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info("Train label distribution: %s", y_train.value_counts().to_dict())

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=False,
    )
    model = LogisticRegression(
        C=c,
        max_iter=1_000,
        class_weight="balanced",
        random_state=42,
    )

    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_val_tfidf = vectorizer.transform(x_val)
    x_test_tfidf = vectorizer.transform(x_test)

    logger.info(
        "Vectorizer | vocab_size=%d ngram_range=%s max_features=%d",
        len(vectorizer.vocabulary_),
        ngram_range,
        max_features,
    )

    model.fit(x_train_tfidf, y_train)

    val_pred = pd.Series(model.predict(x_val_tfidf))
    test_pred = pd.Series(model.predict(x_test_tfidf))

    val_metrics = evaluate_split(y_val, val_pred)
    test_metrics = evaluate_split(y_test, test_pred)

    logger.info(
        "Validation | accuracy=%.4f macro_f1=%.4f",
        val_metrics.accuracy,
        val_metrics.macro_f1,
    )
    logger.info(
        "Test | accuracy=%.4f macro_f1=%.4f",
        test_metrics.accuracy,
        test_metrics.macro_f1,
    )

    return vectorizer, model, val_metrics, test_metrics
