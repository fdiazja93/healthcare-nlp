from __future__ import annotations

import argparse
import logging
from pathlib import Path

from healthcare_nlp.baseline import run_tfidf_logreg_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TF-IDF + Logistic Regression baseline"
    )

    parser.add_argument(
        "--path-to-splits",
        type=Path,
        default=Path("data/processed"),
        help="Path to preprocessed dataset splits",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    _, _, val_metrics, test_metrics = run_tfidf_logreg_baseline(args.path_to_splits)

    logging.info("Validation macro_f1: %.4f", val_metrics.macro_f1)
    logging.info("Test macro_f1: %.4f", test_metrics.macro_f1)


if __name__ == "__main__":
    main()
