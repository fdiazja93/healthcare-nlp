from __future__ import annotations

import argparse
import logging
from pathlib import Path

from healthcare_nlp.data import BertBatch, build_dataloaders
from healthcare_nlp.model import get_tokenizer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create dataloaders for training, validation, and test sets"
    )

    parser.add_argument(
        "--path-to-splits",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed dataset splits",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the DataLoaders",
    )

    return parser.parse_args()


def _log_batch(name: str, batch: BertBatch, n: int = 2) -> None:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    logger.info(
        "%s batch | input_ids=%s attention_mask=%s labels=%s",
        name,
        tuple(input_ids.shape),
        tuple(attention_mask.shape),
        tuple(labels.shape),
    )

    # Inspect first n samples (truncated sequences)
    logger.info(
        "%s batch | input_ids_sample=%s",
        name,
        input_ids[:n, :10].tolist(),  # first n samples, first 10 tokens
    )
    logger.info(
        "%s batch | attention_mask_sample=%s",
        name,
        attention_mask[:n, :10].tolist(),
    )
    logger.info(
        "%s batch | labels_sample=%s",
        name,
        labels[:n].tolist(),
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("Starting dataloader CLI")
    logger.info("Arguments | path=%s batch_size=%d", args.path_to_splits, args.batch_size)

    tokenizer = get_tokenizer()
    logger.info("Tokenizer loaded")

    train_loader, val_loader, test_loader = build_dataloaders(
        path_to_splits=args.path_to_splits,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    logger.info(
        "Dataloaders ready | train_batches=%d val_batches=%d test_batches=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    train_batch: BertBatch = next(iter(train_loader))
    val_batch: BertBatch = next(iter(val_loader))
    test_batch: BertBatch = next(iter(test_loader))

    _log_batch("Train", train_batch)
    _log_batch("Validation", val_batch)
    _log_batch("Test", test_batch)


if __name__ == "__main__":
    main()
