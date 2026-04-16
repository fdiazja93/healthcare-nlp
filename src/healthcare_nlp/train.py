"""Training and evaluation loops for BERT fine-tuning with W&B logging."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import wandb
import torch
import torch.nn as nn
from sklearn.metrics import f1_score as sklearn_f1_score
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from healthcare_nlp.data import BertBatch


logger = logging.getLogger(__name__)


_DEFAULT_TRAINING_PARAMS = {
    "epochs": 5,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_patience": 1,
    "lr_factor": 0.1,
}

_REQUIRED_TRAINING_PARAMS = set(_DEFAULT_TRAINING_PARAMS.keys())


class EarlyStopping:
    """Stops training after `patience` epochs without validation loss improvement."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self._best_loss = float("inf")
        self._counter = 0

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @property
    def counter(self) -> int:
        return self._counter

    def step(self, val_loss: float) -> bool:
        if val_loss < self._best_loss:
            self._best_loss = val_loss
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience


def move_batch_to_device(batch: BertBatch, device: torch.device) -> BertBatch:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


def _forward_model(model: nn.Module, batch: BertBatch) -> dict[str, Tensor]:
    outputs = model(**batch)
    return {
        "loss": cast(Tensor, outputs.loss),
        "logits": cast(Tensor, outputs.logits),
    }


@torch.no_grad()
def evaluate_bert(
    model: nn.Module,
    loader: DataLoader[BertBatch],
    device: torch.device,
    class_weights: Tensor | None = None,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Evaluate the model and return (average loss, macro-F1)."""
    model.eval()

    total_loss = 0.0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = move_batch_to_device(batch, device)

        outputs = _forward_model(model, batch)
        logits = outputs["logits"]
        labels = batch["labels"]

        if class_weights is not None:
            loss = nn.functional.cross_entropy(logits, labels, weight=class_weights)
        else:
            loss = outputs["loss"]

        predictions = torch.argmax(logits, dim=-1)

        total_loss += loss.item()
        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

    batches_seen = min(max_batches, len(loader)) if max_batches is not None else len(loader)
    avg_loss = total_loss / batches_seen
    macro_f1 = float(sklearn_f1_score(all_labels, all_predictions, average="macro"))

    return avg_loss, macro_f1


def train_bert(
    model: nn.Module,
    train_loader: DataLoader[BertBatch],
    val_loader: DataLoader[BertBatch],
    device: torch.device,
    training_params: dict | None = None,
    class_weights: Tensor | None = None,
    max_batches: int | None = None,
    patience: int | None = None,
    checkpoint_path: Path | None = None,
) -> None:
    """Fine-tune BERT with AdamW, linear warmup + ReduceLROnPlateau, and optional early stopping."""
    params = training_params if training_params is not None else _DEFAULT_TRAINING_PARAMS

    missing = _REQUIRED_TRAINING_PARAMS - set(params.keys())
    if missing:
        raise ValueError(f"training_params is missing required keys: {missing}")

    lr_patience: int = params["lr_patience"]
    if patience is not None and patience <= lr_patience:
        raise ValueError(
            f"Early stopping patience ({patience}) must be greater than lr_patience ({lr_patience}). "
            f"The model needs time to respond to LR reductions before stopping."
        )

    lr = params["lr"]
    weight_decay = params["weight_decay"]
    num_epochs = params["epochs"]
    warmup_ratio = params["warmup_ratio"]
    lr_factor = params["lr_factor"]

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    batches_per_epoch = min(max_batches, len(train_loader)) if max_batches is not None else len(train_loader)
    warmup_steps = int(warmup_ratio * num_epochs * batches_per_epoch)

    def warmup_lambda(current_step: int) -> float:
        if warmup_steps == 0:
            return 1.0
        return min(1.0, current_step / warmup_steps)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience)

    logger.info(
        "Starting training | epochs=%d | lr=%.2e | weight_decay=%.4f | warmup_steps=%d | lr_patience=%d | lr_factor=%.2f | device=%s",
        num_epochs,
        lr,
        weight_decay,
        warmup_steps,
        lr_patience,
        lr_factor,
        device,
    )

    early_stopping = EarlyStopping(patience=patience) if patience is not None else None
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0.0
        all_train_predictions: list[int] = []
        all_train_labels: list[int] = []

        for step, batch in enumerate(train_loader, start=1):
            if max_batches is not None and step > max_batches:
                break
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad()

            outputs = _forward_model(model, batch)
            logits = outputs["logits"]
            labels = batch["labels"]

            if class_weights is not None:
                loss = nn.functional.cross_entropy(logits, labels, weight=class_weights)
            else:
                loss = outputs["loss"]

            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step <= warmup_steps:
                warmup_scheduler.step()

            predictions = torch.argmax(logits, dim=-1)

            total_train_loss += loss.item()
            all_train_predictions.extend(predictions.tolist())
            all_train_labels.extend(labels.tolist())

            if step % 50 == 0 or step == len(train_loader):
                logger.info(
                    "Epoch %d/%d | Step %d/%d | Train Loss %.4f",
                    epoch + 1,
                    num_epochs,
                    step,
                    len(train_loader),
                    total_train_loss / step,
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/loss_step": total_train_loss / step,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

        steps_taken = min(max_batches, len(train_loader)) if max_batches is not None else len(train_loader)
        avg_train_loss = total_train_loss / steps_taken
        train_f1 = float(sklearn_f1_score(all_train_labels, all_train_predictions, average="macro"))

        val_loss, val_f1 = evaluate_bert(
            model=model,
            loader=val_loader,
            device=device,
            class_weights=class_weights,
            max_batches=max_batches,
        )

        if global_step > warmup_steps:
            plateau_scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %d/%d completed | train_loss=%.4f | train_f1=%.4f | val_loss=%.4f | val_f1=%.4f | lr=%.2e",
            epoch + 1,
            num_epochs,
            avg_train_loss,
            train_f1,
            val_loss,
            val_f1,
            current_lr,
        )

        if wandb.run is not None:
            wandb.log(
                {
                    "train/loss": avg_train_loss,
                    "train/f1": train_f1,
                    "val/loss": val_loss,
                    "val/f1": val_f1,
                },
                step=global_step,
            )

        if checkpoint_path is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("Checkpoint saved | path=%s | val_loss=%.4f", checkpoint_path, val_loss)

        if early_stopping is not None and early_stopping.step(val_loss):
            logger.info(
                "Early stopping triggered | patience=%d | best_val_loss=%.4f",
                patience,
                early_stopping.best_loss,
            )
            break
