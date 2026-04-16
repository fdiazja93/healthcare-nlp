from __future__ import annotations

from typing import cast

import pytest
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

from healthcare_nlp.model import configure_trainable_layers, freeze_bert_bulk, unfreeze_n_last_layers


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _FakeBertEncoder(nn.Module):
    def __init__(self, num_layers: int = 4) -> None:
        super().__init__()
        self.layer = nn.ModuleList([_FakeLayer() for _ in range(num_layers)])


class _FakeBert(nn.Module):
    def __init__(self, num_layers: int = 4) -> None:
        super().__init__()
        self.encoder = _FakeBertEncoder(num_layers)


class _FakeModel(nn.Module):
    def __init__(self, num_layers: int = 4) -> None:
        super().__init__()
        self.bert = _FakeBert(num_layers)
        self.classifier = nn.Linear(8, 2)


def _make_model(num_layers: int = 4) -> BertForSequenceClassification:
    return cast(BertForSequenceClassification, _FakeModel(num_layers=num_layers))


def test_freeze_bert_bulk_freezes_all_bert_params() -> None:
    model = _make_model()
    freeze_bert_bulk(model)

    assert all(not p.requires_grad for p in model.bert.parameters())


def test_freeze_bert_bulk_leaves_classifier_trainable() -> None:
    model = _make_model()
    freeze_bert_bulk(model)

    assert all(p.requires_grad for p in model.classifier.parameters())


def test_unfreeze_n_last_layers_unfreezes_correct_layers() -> None:
    model = _make_model(num_layers=4)
    freeze_bert_bulk(model)
    unfreeze_n_last_layers(model, n_layers=2)

    layers = model.bert.encoder.layer
    assert all(not p.requires_grad for p in layers[0].parameters())
    assert all(not p.requires_grad for p in layers[1].parameters())
    assert all(p.requires_grad for p in layers[2].parameters())
    assert all(p.requires_grad for p in layers[3].parameters())


def test_unfreeze_n_last_layers_all_unfreezes_everything() -> None:
    model = _make_model(num_layers=4)
    freeze_bert_bulk(model)
    unfreeze_n_last_layers(model, n_layers=4)

    assert all(p.requires_grad for p in model.bert.encoder.layer.parameters())


def test_configure_trainable_layers_head_only_freezes_bert() -> None:
    model = _make_model()
    configure_trainable_layers(model, mode="head_only")

    assert all(not p.requires_grad for p in model.bert.parameters())


def test_configure_trainable_layers_full_training_leaves_all_trainable() -> None:
    model = _make_model()
    configure_trainable_layers(model, mode="full_training")

    assert all(p.requires_grad for p in model.parameters())


def test_configure_trainable_layers_top_n_freezes_lower_unfreezes_upper() -> None:
    model = _make_model(num_layers=4)
    configure_trainable_layers(model, mode="top_n", n_layers=2)

    layers = model.bert.encoder.layer
    assert all(not p.requires_grad for p in layers[0].parameters())
    assert all(not p.requires_grad for p in layers[1].parameters())
    assert all(p.requires_grad for p in layers[2].parameters())
    assert all(p.requires_grad for p in layers[3].parameters())


def test_configure_trainable_layers_top_n_requires_n_layers() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="n_layers"):
        configure_trainable_layers(model, mode="top_n", n_layers=None)


def test_configure_trainable_layers_invalid_mode_raises() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="not supported"):
        configure_trainable_layers(model, mode="unknown")
