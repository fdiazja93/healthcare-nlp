from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import wandb
import torch

from healthcare_nlp.data import build_dataloaders, compute_class_weights, load_preprocessed_splits
from healthcare_nlp.model import configure_trainable_layers, get_tokenizer, load_bert
from healthcare_nlp.train import train_bert

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT on the healthcare NLP classification task"
    )

    parser.add_argument(
        "--path-to-splits",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed dataset splits",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the DataLoaders",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for AdamW optimizer",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer",
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of training steps used for linear warmup",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: run 1 epoch with 10 train batches and 5 eval batches (smoke test)",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (epochs without val loss improvement). Disabled if not set.",
    )

    parser.add_argument(
        "--lr-patience",
        type=int,
        default=1,
        help="ReduceLROnPlateau patience (epochs without val loss improvement before reducing LR).",
    )

    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.1,
        help="ReduceLROnPlateau multiplicative factor for LR reduction.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to save the best model checkpoint. Defaults to <output-dir>/best.pt.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the final model and tokenizer. Defaults to models/<group>/<run-name>.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="W&B run name (e.g. 'stage1-lr1e-3', 'stage2-top4').",
    )

    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="W&B run group for clustering related experiments (e.g. 'stage1', 'stage2').",
    )

    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Weight the loss by inverse class frequency to handle class imbalance.",
    )

    parser.add_argument(
        "--training-mode",
        default="head_only",
        choices=["head_only", "full_training", "top_n"],
        help="Which layers to train: 'head_only' (classifier only), 'full_training' (all), or 'top_n' (last n encoder layers + head).",
    )

    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Number of top encoder layers to unfreeze when --training-mode=top_n.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--model-endpoint",
        type=str,
        default="google-bert/bert-base-uncased",
        help="HuggingFace model endpoint.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    set_seed(args.seed)
    logger.info("Seed set | %d", args.seed)

    base = Path("models")
    run_dir = base / args.group / args.run_name if args.group else base / args.run_name

    if args.output_dir is None:
        args.output_dir = run_dir

    if args.checkpoint_path is None:
        args.checkpoint_path = run_dir / "best.pt"

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training CLI")
    logger.info(
        "Arguments | path=%s batch_size=%d epochs=%d lr=%.2e weight_decay=%.4f warmup_ratio=%.2f use_class_weights=%s training_mode=%s n_layers=%s model_endpoint=%s",
        args.path_to_splits,
        args.batch_size,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.warmup_ratio,
        args.use_class_weights,
        args.training_mode,
        args.n_layers,
        args.model_endpoint,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device | %s", device)

    tokenizer = get_tokenizer(endpoint=args.model_endpoint)
    logger.info("Tokenizer loaded")

    fast_mode = args.fast
    max_batches: int | None = 10 if fast_mode else None
    if fast_mode:
        logger.info("Fast mode enabled — 1 epoch, 10 train batches, 10 eval batches")

    train_df, val_df, test_df = load_preprocessed_splits(args.path_to_splits)

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_df)
        class_weights = class_weights.to(device)

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

    model = load_bert(endpoint=args.model_endpoint)
    configure_trainable_layers(model, mode=args.training_mode, n_layers=args.n_layers)
    model.to(device)  # type: ignore

    training_params = {
        "epochs": 1 if fast_mode else args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_patience": args.lr_patience,
        "lr_factor": args.lr_factor,
    }

    wandb.init(
        project="healthcare-nlp",
        name=args.run_name,
        group=args.group,
        config={
            **training_params,
            "batch_size": args.batch_size,
            "training_mode": args.training_mode,
            "n_layers": args.n_layers,
            "use_class_weights": args.use_class_weights,
            "patience": args.patience,
            "fast_mode": fast_mode,
            "seed": args.seed,
            "device": str(device),
            "model_endpoint": args.model_endpoint,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "test_batches": len(test_loader),
            "train_pos_ratio": float(train_df["label"].mean().item()),
        },
    )

    train_bert(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        training_params=training_params,
        class_weights=class_weights,
        max_batches=max_batches,
        patience=args.patience,
        checkpoint_path=args.checkpoint_path,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)  # type: ignore
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Model and tokenizer saved | path=%s", args.output_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
