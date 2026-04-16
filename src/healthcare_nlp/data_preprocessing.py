"""Load, clean, and split the ADE corpus into train/val/test parquet files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pandas as pd
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from sklearn.model_selection import train_test_split

from healthcare_nlp.text_utils import normalize_text

logger = logging.getLogger(__name__)

_COL_TEXT = "text"
_COL_LABEL = "label"
_COL_LABEL_TEXT = "label_text"
_COL_NORM_TEXT = "_norm_text"
_REQUIRED_COLS = {_COL_TEXT, _COL_LABEL, _COL_LABEL_TEXT}
_EXPECTED_LABELS = {0, 1}


def _validate_columns(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _validate_labels(df: pd.DataFrame) -> None:
    labels = set(df[_COL_LABEL].dropna().unique().tolist())
    if not labels.issubset(_EXPECTED_LABELS):
        raise ValueError(f"Unexpected label values: {sorted(labels)}")


def _validate_ready_for_cleaning(df: pd.DataFrame) -> None:
    _validate_columns(df)
    _validate_labels(df)


def _validate_no_overlap(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    overlap = set(df_train[_COL_NORM_TEXT].dropna().unique()).intersection(
        set(df_test[_COL_NORM_TEXT].dropna().unique())
    )
    if overlap:
        raise RuntimeError("Train/test overlap still present after cleaning")


def load_ade_dataset() -> DatasetDict:
    """Download the ADE corpus v2 binary classification dataset from the Hub."""
    logger.info("Loading ADE dataset")
    return load_dataset("SetFit/ade_corpus_v2_classification")


def load_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the raw train/test splits as pandas DataFrames after schema checks."""
    ds = load_ade_dataset()

    train_pd = ds["train"].to_pandas()
    df_train = (
        train_pd
        if isinstance(train_pd, pd.DataFrame)
        else pd.concat(list(train_pd), ignore_index=True)
    )

    test_pd = ds["test"].to_pandas()
    df_test = (
        test_pd
        if isinstance(test_pd, pd.DataFrame)
        else pd.concat(list(test_pd), ignore_index=True)
    )

    _validate_ready_for_cleaning(df_train)
    _validate_ready_for_cleaning(df_test)

    logger.info("Loaded raw splits: train=%d test=%d", len(df_train), len(df_test))

    return df_train, df_test


def add_normalized_text(df_in: pd.DataFrame, col: str = _COL_TEXT) -> pd.DataFrame:
    """Attach a `_norm_text` column used as the canonical key for dedupe/overlap checks."""
    _validate_ready_for_cleaning(df_in)
    df = df_in.copy()
    df[_COL_NORM_TEXT] = df[col].astype(str).map(normalize_text)
    return df


def drop_label_conflicts(df_in: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where the same normalized text appears with more than one label."""
    if _COL_NORM_TEXT not in df_in.columns:
        raise ValueError(f"Missing required column: {_COL_NORM_TEXT}")

    df = df_in.copy()
    n_before = len(df)
    conflict_mask = df.groupby(_COL_NORM_TEXT)[_COL_LABEL].transform("nunique").gt(1)
    df = df.loc[~conflict_mask].reset_index(drop=True)
    logger.info("Dropped %d rows with label conflicts", n_before - len(df))
    return df


def dedupe_within_split(df_in: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate normalized rows inside a single split."""
    if _COL_NORM_TEXT not in df_in.columns:
        raise ValueError(f"Missing required column: {_COL_NORM_TEXT}")

    df = df_in.copy()
    n_before = len(df)
    df = df.drop_duplicates(subset=[_COL_NORM_TEXT]).reset_index(drop=True)
    logger.info("Dropped %d duplicate rows within split", n_before - len(df))
    return df


def remove_train_test_overlap(
    df_train_in: pd.DataFrame,
    df_test_in: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove any training rows whose normalized text also appears in the test split."""
    if _COL_NORM_TEXT not in df_train_in.columns or _COL_NORM_TEXT not in df_test_in.columns:
        raise ValueError(f"Missing required column: {_COL_NORM_TEXT}")

    df_train = df_train_in.copy()
    df_test = df_test_in.copy()

    n_before = len(df_train)
    test_texts = df_test[_COL_NORM_TEXT].dropna().unique().tolist()
    df_train = df_train.loc[
        ~df_train[_COL_NORM_TEXT].isin(test_texts)
    ].reset_index(drop=True)

    _validate_no_overlap(df_train, df_test)

    logger.info("Removed %d overlapping train rows", n_before - len(df_train))

    return df_train, df_test


def train_val_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation split with a fixed seed for reproducibility."""
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")

    _validate_ready_for_cleaning(df)

    label_counts = df[_COL_LABEL].value_counts()
    n_classes = len(label_counts)
    n_samples = len(df)

    if n_classes < 2:
        raise ValueError("At least two classes are required for stratified splitting")

    if label_counts.min() < 2:
        raise ValueError(
            "Each class must have at least two samples for stratified splitting"
        )

    n_val = max(1, int(round(n_samples * val_size)))
    n_train = n_samples - n_val

    if n_val < n_classes or n_train < n_classes:
        raise ValueError(
            "Dataset is too small for stratified splitting with the requested val_size"
        )

    df_train, df_val = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            df,
            test_size=val_size,
            stratify=df[_COL_LABEL],
            random_state=42,
        ),
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    logger.info(
        "Created train/validation split: train=%d val=%d",
        len(df_train),
        len(df_val),
    )

    return df_train, df_val


def clean_splits(val_size: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end cleaning pipeline returning (train, val, test) DataFrames."""
    logger.info("Starting data cleaning pipeline")

    df_train, df_test = load_train_test_data()
    df_train = add_normalized_text(df_train)
    df_test = add_normalized_text(df_test)
    df_train = drop_label_conflicts(df_train)
    df_test = drop_label_conflicts(df_test)
    df_train = dedupe_within_split(df_train)
    df_test = dedupe_within_split(df_test)
    df_train, df_test = remove_train_test_overlap(df_train, df_test)
    df_train, df_val = train_val_split(df_train, val_size)

    logger.info(
        "Finished data cleaning pipeline: train=%d val=%d test=%d",
        len(df_train),
        len(df_val),
        len(df_test),
    )

    return df_train, df_val, df_test


def save_clean_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    data_dir: Path,
) -> None:
    """Persist the cleaned splits as parquet files under `data_dir`."""
    data_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(data_dir / "train.parquet", index=False)
    df_val.to_parquet(data_dir / "val.parquet", index=False)
    df_test.to_parquet(data_dir / "test.parquet", index=False)

    logger.info("Saved processed splits to %s", data_dir)
