from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import torch
from transformers import PreTrainedTokenizerBase

from healthcare_nlp.data import (
    PubMedDataset,
    BertBatch,
    BertSample,
    get_collate_fn,
    get_loader,
    load_preprocessed_splits,
)


class DummyTokenizer:
    def __call__(
        self,
        text: str,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        tokens = text.split()
        input_ids = list(range(1, len(tokens) + 1))
        attention_mask = [1] * len(input_ids)

        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def pad(
        self,
        features: list[BertSample],
        _padding: bool = True,
        _return_tensors: str | None = None,
        **kwargs,
    ) -> BertBatch:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            ids = list(f["input_ids"])
            mask = list(f["attention_mask"])
            pad_len = max_len - len(ids)

            input_ids.append(ids + [0] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(int(f["labels"]))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_df(
    texts: list[str] | None = None,
    labels: list[int] | None = None,
) -> pd.DataFrame:
    if texts is None:
        texts = ["short text", "a bit longer text", "tiny"]
    if labels is None:
        labels = [0, 1, 0]

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
    train_df.to_parquet(path / "train.parquet", index=False)
    val_df.to_parquet(path / "val.parquet", index=False)
    test_df.to_parquet(path / "test.parquet", index=False)


def test_load_preprocessed_splits_returns_three_dataframes(tmp_path: Path) -> None:
    df = make_df()
    write_splits(tmp_path, df, df, df)

    train_df, val_df, test_df = load_preprocessed_splits(tmp_path)

    assert len(train_df) == 3
    assert len(val_df) == 3
    assert len(test_df) == 3


def test_load_preprocessed_splits_raises_for_missing_columns(tmp_path: Path) -> None:
    good_df = make_df()
    bad_df = pd.DataFrame({"_norm_text": ["a", "b"]})
    write_splits(tmp_path, good_df, bad_df, good_df)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_preprocessed_splits(tmp_path)


def test_dataset_len() -> None:
    df = make_df()
    tokenizer = cast(PreTrainedTokenizerBase, DummyTokenizer())

    ds = PubMedDataset(df, tokenizer=tokenizer)

    assert len(ds) == 3


def test_dataset_getitem_returns_expected_keys_and_types() -> None:
    df = make_df()
    tokenizer = cast(PreTrainedTokenizerBase, DummyTokenizer())


    ds = PubMedDataset(df, tokenizer=tokenizer)
    sample = ds[0]

    assert set(sample.keys()) == {"input_ids", "attention_mask", "labels"}
    assert isinstance(sample["input_ids"], list)
    assert isinstance(sample["attention_mask"], list)
    assert isinstance(sample["labels"], int)


def test_dataset_applies_truncation() -> None:
    df = make_df(
        texts=["one two three four five six"],
        labels=[1],
    )
    tokenizer = cast(PreTrainedTokenizerBase, DummyTokenizer())

    ds = PubMedDataset(df, tokenizer=tokenizer, max_length=3)
    sample = ds[0]

    assert sample["input_ids"] == [1, 2, 3]
    assert sample["attention_mask"] == [1, 1, 1]
    assert sample["labels"] == 1


def test_get_loader_returns_padded_batch() -> None:
    df = make_df(
        texts=["one two", "one two three four"],
        labels=[0, 1],
    )
    tokenizer = cast(PreTrainedTokenizerBase, DummyTokenizer())
    collate_fn = get_collate_fn(tokenizer)

    ds = PubMedDataset(df, tokenizer=tokenizer)
    loader = get_loader(ds, collate_fn=collate_fn, batch_size=2, shuffle=False)

    batch = next(iter(loader))

    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert isinstance(batch["attention_mask"], torch.Tensor)
    assert isinstance(batch["labels"], torch.Tensor)
    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].shape == (2, 4)
    assert batch["labels"].shape == (2,)
