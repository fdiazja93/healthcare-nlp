"""Model and tokenizer factories plus layer-freezing utilities for staged fine-tuning."""
from __future__ import annotations

import logging

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

_MODEL_ENDPOINT = "google-bert/bert-base-uncased"


def get_tokenizer(endpoint: str = _MODEL_ENDPOINT) -> PreTrainedTokenizerBase:
    """Load a HuggingFace tokenizer from an endpoint (Hub id or local path)."""
    tokenizer = AutoTokenizer.from_pretrained(endpoint)
    logger.info("Tokenizer loaded | endpoint=%s", endpoint)
    return tokenizer


def load_bert(endpoint: str = _MODEL_ENDPOINT, num_labels: int = 2) -> BertForSequenceClassification:
    """Load a BERT-for-sequence-classification model with a fresh classification head."""
    model = AutoModelForSequenceClassification.from_pretrained(endpoint, num_labels=num_labels)
    logger.info("Model loaded | endpoint=%s | num_labels=%d", endpoint, num_labels)
    return model


def get_collate_fn(tokenizer: PreTrainedTokenizerBase) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer, padding=True)


def freeze_bert_bulk(model: BertForSequenceClassification) -> None:
    for param in model.bert.parameters():
        param.requires_grad = False
    logger.info("BERT encoder frozen")


def unfreeze_n_last_layers(model: BertForSequenceClassification, n_layers: int) -> None:
    for param in model.bert.encoder.layer[-n_layers:].parameters():
        param.requires_grad = True
    logger.info("Unfroze last %d encoder layers", n_layers)


def configure_trainable_layers(
    model: BertForSequenceClassification,
    mode: str = "head_only",
    n_layers: int | None = None,
) -> None:
    """Configure which parameters receive gradients: head-only, top-n encoder layers, or full."""
    if mode == "top_n" and n_layers is None:
        raise ValueError("n_layers must be provided when mode='top_n'")
    if mode == "full_training":
        pass
    elif mode == "top_n" and n_layers is not None:
        freeze_bert_bulk(model)
        unfreeze_n_last_layers(model, n_layers)
    elif mode == "head_only":
        freeze_bert_bulk(model)
    else:
        raise ValueError(f"{mode} is not supported. The mode has to be in [head_only, full_training, top_n]")
    logger.info("Trainable layers configured | mode=%s | n_layers=%s", mode, n_layers)


