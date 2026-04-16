from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from healthcare_nlp.baseline import (
    SplitMetrics,
    _validate_columns,
    evaluate_split,
    load_preprocessed_splits,
    run_tfidf_logreg_baseline,
)


def make_df(
    texts: list[str] | None = None,
    labels: list[int] | None = None,
) -> pd.DataFrame:
    if texts is None:
        texts = ["patient had rash", "treatment was well tolerated"]
    if labels is None:
        labels = [1, 0]

    return pd.DataFrame(
        {
            "_norm_text": texts,
            "label": labels,
        }
    )


def write_splits(
    path: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_df.to_parquet(path / "train.parquet")
    val_df.to_parquet(path / "val.parquet")
    test_df.to_parquet(path / "test.parquet")


def test_validate_columns_accepts_required_columns() -> None:
    df = make_df()
    _validate_columns(df)


def test_validate_columns_raises_for_missing_columns() -> None:
    df = pd.DataFrame({"_norm_text": ["a", "b"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        _validate_columns(df)


def test_load_preprocessed_splits_returns_three_dataframes(tmp_path: Path) -> None:
    df = make_df()
    write_splits(tmp_path, df, df, df)

    train_df, val_df, test_df = load_preprocessed_splits(tmp_path)

    assert list(train_df.columns) == ["_norm_text", "label"]
    assert list(val_df.columns) == ["_norm_text", "label"]
    assert list(test_df.columns) == ["_norm_text", "label"]
    assert len(train_df) == 2
    assert len(val_df) == 2
    assert len(test_df) == 2


def test_load_preprocessed_splits_raises_if_required_columns_are_missing(
    tmp_path: Path,
) -> None:
    good_df = make_df()
    bad_df = pd.DataFrame({"_norm_text": ["text 1", "text 2"]})
    write_splits(tmp_path, good_df, bad_df, good_df)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_preprocessed_splits(tmp_path)


def test_evaluate_split_returns_expected_metrics() -> None:
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0])

    metrics = evaluate_split(y_true, y_pred)

    assert metrics.accuracy == 0.75
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert isinstance(metrics.report, str)
    assert "precision" in metrics.report
    assert "recall" in metrics.report


def test_run_tfidf_logreg_baseline_runs_end_to_end(tmp_path: Path) -> None:
    train_df = make_df(
        texts=[
            "patient had rash",
            "patient had headache",
            "no adverse event observed",
            "treatment was well tolerated",
        ],
        labels=[1, 1, 0, 0],
    )
    val_df = make_df(
        texts=[
            "rash after medication",
            "well tolerated treatment",
        ],
        labels=[1, 0],
    )
    test_df = make_df(
        texts=[
            "headache after drug",
            "no adverse event",
        ],
        labels=[1, 0],
    )

    write_splits(tmp_path, train_df, val_df, test_df)

    vectorizer, model, val_metrics, test_metrics = run_tfidf_logreg_baseline(tmp_path)

    assert hasattr(vectorizer, "vocabulary_")
    assert len(vectorizer.vocabulary_) > 0

    assert hasattr(model, "classes_")
    assert set(model.classes_.tolist()) == {0, 1}

    assert isinstance(val_metrics, SplitMetrics)
    assert isinstance(test_metrics, SplitMetrics)

    assert 0.0 <= val_metrics.accuracy <= 1.0
    assert 0.0 <= val_metrics.macro_f1 <= 1.0
    assert 0.0 <= test_metrics.accuracy <= 1.0
    assert 0.0 <= test_metrics.macro_f1 <= 1.0
