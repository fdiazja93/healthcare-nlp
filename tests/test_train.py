from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from healthcare_nlp.data import BertBatch
from healthcare_nlp.train import EarlyStopping, evaluate_bert, move_batch_to_device, train_bert

_FULL_TRAINING_PARAMS = {
    "epochs": 1,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_patience": 1,
    "lr_factor": 0.1,
}


class DummyBatchDataset(Dataset[BertBatch]):
    def __init__(self, batches: list[BertBatch]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> BertBatch:
        return self.batches[index]


class DummyClassifier(nn.Module):
    def __init__(self, input_dim: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        logits = self.linear(input_ids.float())
        loss = nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits)


def make_loader(batches: list[BertBatch]) -> DataLoader[BertBatch]:
    ds = DummyBatchDataset(batches)
    return DataLoader(ds, batch_size=None, shuffle=False)


def test_move_batch_to_device_moves_tensors_only() -> None:
    batch: BertBatch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }

    moved = move_batch_to_device(batch, torch.device("cpu"))

    assert moved is not batch
    assert moved["input_ids"].device.type == "cpu"
    assert moved["attention_mask"].device.type == "cpu"
    assert moved["labels"].device.type == "cpu"


def test_evaluate_bert_returns_expected_loss_and_accuracy() -> None:
    class FixedEvalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.eval_called = False

        def eval(self) -> nn.Module:
            self.eval_called = True
            return super().eval()

        def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor,
            labels: Tensor,
        ) -> SimpleNamespace:
            del input_ids
            del attention_mask

            if torch.equal(labels, torch.tensor([0, 1], dtype=torch.long)):
                logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
                loss = torch.tensor(0.4)
            else:
                logits = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
                loss = torch.tensor(0.8)

            return SimpleNamespace(loss=loss, logits=logits)

    loader = make_loader(
        [
            {
                "input_ids": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
                "labels": torch.tensor([0, 1], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
                "labels": torch.tensor([1, 0], dtype=torch.long),
            },
        ]
    )

    model = FixedEvalModel()

    loss, f1 = evaluate_bert(model=model, loader=loader, device=torch.device("cpu"))

    assert model.eval_called is True
    assert loss == pytest.approx(0.6)
    assert f1 == pytest.approx(1.0)


def test_evaluate_bert_respects_max_batches() -> None:
    batches_seen: list[int] = []

    class CountingModel(nn.Module):
        def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor,
            labels: Tensor,
        ) -> SimpleNamespace:
            batches_seen.append(1)
            logits = torch.tensor([[5.0, 0.0]])
            loss = torch.tensor(0.5)
            return SimpleNamespace(loss=loss, logits=logits)

    loader = make_loader(
        [
            {
                "input_ids": torch.tensor([[1.0, 0.0]]),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "labels": torch.tensor([0], dtype=torch.long),
            }
            for _ in range(5)
        ]
    )

    evaluate_bert(model=CountingModel(), loader=loader, device=torch.device("cpu"), max_batches=2)

    assert len(batches_seen) == 2


def test_train_bert_respects_max_batches() -> None:
    steps_per_epoch: list[int] = []

    class CountingClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor,
            labels: Tensor,
        ) -> SimpleNamespace:
            if self.training:
                steps_per_epoch.append(1)
            logits = self.linear(input_ids.float())
            loss = nn.functional.cross_entropy(logits, labels)
            return SimpleNamespace(loss=loss, logits=logits)

    single_batch: BertBatch = {
        "input_ids": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([0], dtype=torch.long),
    }
    train_loader = make_loader([single_batch] * 5)
    val_loader = make_loader([single_batch])

    train_bert(
        model=CountingClassifier(),
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        training_params=_FULL_TRAINING_PARAMS,
        max_batches=2,
    )

    assert len(steps_per_epoch) == 2


def test_train_bert_runs_one_epoch() -> None:
    model = DummyClassifier()

    train_loader = make_loader(
        [
            {
                "input_ids": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
                "labels": torch.tensor([0], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
                "labels": torch.tensor([1], dtype=torch.long),
            },
        ]
    )

    val_loader = make_loader(
        [
            {
                "input_ids": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
                "labels": torch.tensor([0], dtype=torch.long),
            }
        ]
    )

    train_bert(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        training_params=_FULL_TRAINING_PARAMS,
    )


def test_early_stopping_triggers_after_patience_epochs() -> None:
    es = EarlyStopping(patience=2)
    assert es.step(1.0) is False
    assert es.step(1.1) is False
    assert es.step(1.2) is True


def test_early_stopping_resets_counter_on_improvement() -> None:
    es = EarlyStopping(patience=2)
    assert es.step(1.0) is False
    assert es.step(1.1) is False
    assert es.step(0.5) is False
    assert es.counter == 0
    assert es.step(0.6) is False
    assert es.step(0.7) is True



def test_train_bert_stops_early() -> None:
    epochs_run: list[int] = []

    class StallingClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor,
            labels: Tensor,
        ) -> SimpleNamespace:
            logits = self.linear(input_ids.float())
            if self.training:
                epochs_run.append(1)
                loss = nn.functional.cross_entropy(logits, labels)
            else:
                loss = torch.tensor(1.0)
            return SimpleNamespace(loss=loss, logits=logits)

    single_batch: BertBatch = {
        "input_ids": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([0], dtype=torch.long),
    }
    loader = make_loader([single_batch])

    train_bert(
        model=StallingClassifier(),
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        training_params={**_FULL_TRAINING_PARAMS, "epochs": 10},
        patience=2,
    )

    assert len(epochs_run) < 10


def test_train_bert_saves_checkpoint_on_improvement(tmp_path) -> None:
    model = DummyClassifier()
    checkpoint = tmp_path / "best.pt"

    single_batch: BertBatch = {
        "input_ids": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([0], dtype=torch.long),
    }
    loader = make_loader([single_batch])

    train_bert(
        model=model,
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        training_params={**_FULL_TRAINING_PARAMS, "epochs": 2},
        checkpoint_path=checkpoint,
    )

    assert checkpoint.exists()
    state = torch.load(checkpoint, weights_only=True)
    assert set(state.keys()) == set(model.state_dict().keys())


def test_lr_reduces_after_plateau() -> None:
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=1)

    initial_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(1.0)  # sets best
    scheduler.step(1.0)  # num_bad_epochs = 1
    scheduler.step(1.0)  # num_bad_epochs = 2 > patience=1 → reduce

    assert optimizer.param_groups[0]["lr"] < initial_lr


def test_lr_warms_up_from_zero() -> None:
    from torch.optim.lr_scheduler import LambdaLR

    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    warmup_steps = 5

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps))

    lrs = []
    for _ in range(warmup_steps + 1):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[0] == pytest.approx(0.0)
    assert lrs[-1] == pytest.approx(1e-3)
    assert lrs == sorted(lrs)


def test_train_bert_accepts_lr_scheduling_params() -> None:
    model = DummyClassifier()
    single_batch: BertBatch = {
        "input_ids": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([0], dtype=torch.long),
    }
    loader = make_loader([single_batch])

    train_bert(
        model=model,
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        training_params={**_FULL_TRAINING_PARAMS, "epochs": 3, "warmup_ratio": 0.3, "lr_patience": 1, "lr_factor": 0.5},
    )
