from __future__ import annotations

import logging
from pathlib import Path

import torch
import wandb

from healthcare_nlp.data import build_dataloaders, load_preprocessed_splits
from healthcare_nlp.model import get_tokenizer, load_bert
from healthcare_nlp.train import evaluate_bert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_MODEL_PATH = Path("models/best")
_DATA_PATH = Path("data/processed")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

logger.info("Device | %s", device)

tokenizer = get_tokenizer(endpoint=str(_MODEL_PATH))
model = load_bert(endpoint=str(_MODEL_PATH))
model.to(device)  # type: ignore

_, _, test_df = load_preprocessed_splits(_DATA_PATH)

_, _, test_loader = build_dataloaders(
    path_to_splits=_DATA_PATH,
    tokenizer=tokenizer,
    batch_size=16,
)

wandb.init(
    project="healthcare-nlp",
    name="test-set",
    group="final-evaluation",
)

test_loss, test_f1 = evaluate_bert(
    model=model,
    loader=test_loader,
    device=device,
)

logger.info("Test results | loss=%.4f | macro_f1=%.4f", test_loss, test_f1)

wandb.log({"test/loss": test_loss, "test/f1": test_f1})
wandb.finish()
