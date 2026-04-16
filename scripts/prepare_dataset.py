from __future__ import annotations

import argparse
import logging
from pathlib import Path

from healthcare_nlp.data_preprocessing import clean_splits, save_clean_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ADE dataset splits.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed parquet splits will be saved.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation split fraction for the cleaned training set.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    train, val, test = clean_splits(val_size=args.val_size)
    save_clean_splits(train, val, test, args.output_dir)


if __name__ == "__main__":
    main()
