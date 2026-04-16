"""PyTorch dataset, collator, and DataLoader factory for the BERT fine-tuning pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

_COL_TEXT = "_norm_text"
_COL_LABEL = "label"
_REQUIRED_COLS = {_COL_TEXT, _COL_LABEL}

logger = logging.getLogger(__name__)


class BertSample(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: int


class BertBatch(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor


def _validate_dataset_columns(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _validate_labels_are_integers(df: pd.DataFrame) -> None:
    if not pd.api.types.is_integer_dtype(df[_COL_LABEL]):
        raise TypeError(
            f"Column '{_COL_LABEL}' must contain integer class ids, "
            "not strings or floats."
        )


def load_preprocessed_splits(
    path_to_splits: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the cleaned parquet splits and validate their schema."""
    train = pd.read_parquet(path_to_splits / "train.parquet")
    val = pd.read_parquet(path_to_splits / "val.parquet")
    test = pd.read_parquet(path_to_splits / "test.parquet")

    for name, df in [("train", train), ("val", val), ("test", test)]:
        _validate_dataset_columns(df)
        _validate_labels_are_integers(df)
        if df.empty:
            raise ValueError(f"{name}.parquet is empty.")

    return train, val, test


def compute_class_weights(train_df: pd.DataFrame) -> Tensor:
    """Return balanced (inverse-frequency) class weights as a float tensor."""
    labels = train_df[_COL_LABEL].to_numpy()
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    logger.info("Class weights | %s", dict(zip(classes.tolist(), weights.round(4).tolist())))
    return torch.tensor(weights, dtype=torch.float32)


class PubMedDataset(Dataset):
    """Tokenizes text samples lazily in `__getitem__`; padding happens in the collator."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        text_col: str = _COL_TEXT,
        label_col: str = _COL_LABEL,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        _validate_dataset_columns(df)
        _validate_labels_are_integers(df)

        self.texts = df[text_col].tolist()
        self.labels = df[label_col].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> BertSample:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")

        text = self.texts[idx]
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = cast(list[int], enc["input_ids"])
        attention_mask = cast(list[int], enc["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


def get_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )


def get_loader(
    dataset: Dataset,
    collate_fn: DataCollatorWithPadding,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_dataloaders(
    path_to_splits: Path,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from the cleaned parquet splits."""
    train_df, val_df, test_df = load_preprocessed_splits(path_to_splits)

    logger.info(
        "Dataset sizes | train=%d val=%d test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info(
        "Dataloaders | batch_size=%d max_length=%d num_workers=%d pin_memory=%s",
        batch_size,
        max_length,
        num_workers,
        pin_memory,
    )

    train_ds = PubMedDataset(train_df, tokenizer=tokenizer, max_length=max_length)
    val_ds = PubMedDataset(val_df, tokenizer=tokenizer, max_length=max_length)
    test_ds = PubMedDataset(test_df, tokenizer=tokenizer, max_length=max_length)

    collate_fn = get_collate_fn(tokenizer)

    train_loader = get_loader(
        train_ds,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = get_loader(
        val_ds,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = get_loader(
        test_ds,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
