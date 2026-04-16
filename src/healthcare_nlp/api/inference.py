"""Model loading and single-example inference used by the FastAPI app."""
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import torch
from healthcare_nlp.api.schemas import PredictionResponse

_LABEL_TO_TEXT = {0: "Not-Related", 1: "Related"}


_tokenizer: PreTrainedTokenizerBase | None = None
_model: PreTrainedModel | None = None

def load_model(model_path: Path) -> None:
    """Populate module-level tokenizer/model singletons from a local HF model directory."""
    global _tokenizer, _model
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForSequenceClassification.from_pretrained(model_path)

@torch.no_grad()
def infer(input_text: str, model_name: str) -> PredictionResponse:
    """Run a single forward pass and return the predicted label, probability, and label text."""
    assert _tokenizer is not None and _model is not None, "Model not loaded"
    tokd = _tokenizer(input_text)
    tokd = {k: torch.tensor(v).unsqueeze(0) for k, v in tokd.items()}
    output = _model(**tokd)
    probs = torch.softmax(output.logits, dim=1)
    label = int(torch.argmax(probs).item())
    probability = probs[0, label].item()
    return PredictionResponse(
        probability=probability, 
        label=label,
        label_text=_LABEL_TO_TEXT[label],
        model_name=model_name,
    )
