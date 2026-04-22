# healthcare-nlp

End-to-end NLP project that fine-tunes BERT to classify biomedical sentences as describing an **Adverse Drug Event (ADE)** or not, and ships the trained model as a containerized FastAPI service deployed on AWS ECS. The project is built as a portfolio piece to demonstrate the full lifecycle of a small ML system: data cleaning, a strong baseline, staged fine-tuning experiments tracked in W&B, training on SageMaker, and online inference behind a REST endpoint.

## Live service

The service is deployed on AWS ECS but is kept stopped to avoid unnecessary cloud costs. To request a demo, contact me at fdiazja@gmail.com, and I will spin it up within a few minutes.

Once live, the service is reachable at:

```
POST https://he-0aada1058ebd4ceebc51cc8efd635648.ecs.us-east-2.on.aws/predict
```

Request body:

```json
{ "input_text": "The patient developed severe rash after taking amoxicillin." }
```

Response body:

```json
{
  "probability": 0.987,
  "label": 1,
  "label_text": "Related",
  "model_name": "model_v0"
}
```

Example:

```
curl -X POST https://he-0aada1058ebd4ceebc51cc8efd635648.ecs.us-east-2.on.aws/predict \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Patient developed hepatotoxicity after taking acetaminophen"}'
```

A `GET /health` endpoint is also exposed for health checks.

## Dataset

[`SetFit/ade_corpus_v2_classification`](https://huggingface.co/datasets/SetFit/ade_corpus_v2_classification) — binary classification (`Related` / `Not-Related`) of whether a sentence describes an adverse drug event. The data is class-imbalanced, which is handled with balanced class weights during training.

The preprocessing pipeline (`src/healthcare_nlp/data_preprocessing.py`) does:

1. Schema and label validation.
2. Text normalization (lowercasing, whitespace collapsing) as a canonical dedupe key.
3. Dropping rows where the same normalized text has conflicting labels.
4. Within-split deduplication.
5. Removing any training rows that leak into the test set.
6. Stratified train/validation split (seeded).
7. Persisting the three splits as parquet files.

## Models

- **Baseline** — TF-IDF (uni+bi-grams) + class-balanced Logistic Regression (`src/healthcare_nlp/baseline.py`). Used as a floor that any fine-tuned BERT must clear.
- **BERT fine-tuning** — `bert-base-uncased` and `dmis-lab/biobert` variants (`src/healthcare_nlp/model.py`, `train.py`). Three training modes are supported:
  - `head_only` — freeze the encoder, train the classification head only.
  - `top_n` — unfreeze the top *n* encoder layers plus the head.
  - `full_training` — unfreeze everything.
- Training uses AdamW, linear warmup followed by `ReduceLROnPlateau`, optional early stopping, macro-F1 as the main metric, and best-checkpoint saving on validation loss.

## Experiments

Stage-based shell scripts in `experiments/` drive the experiment sweeps:

- **Stage 1** — head-only training, LR search.
- **Stage 2** — `top_n` unfreezing with varying depth and LR.
- **Stage 3** — switch the encoder to BioBERT for a domain-specific comparison.

Every run logs train/val loss, macro-F1, learning rate, and run config to **Weights & Biases** (project `healthcare-nlp`). Runs are organized by `group` (stage) and `run-name`.

### W&B in this portfolio

All training curves, hyperparameter sweeps, and comparison plots live in my W&B workspace and are the primary artifact for judging the modeling work. Here is the link to the project in W&B:

```
https://wandb.ai/fdiazja-for-personal-projects/healthcare-nlp?nw=nwuserfdiazja
```

## Serving

- `src/healthcare_nlp/api/app.py` — FastAPI app with a `lifespan` hook that syncs the serialized model from S3 into the container at startup, structured JSON access logs with per-request latency, and input/prediction fields attached via a middleware.
- `src/healthcare_nlp/api/inference.py` — tokenization + single forward pass, wrapped in `torch.no_grad()`.
- `docker/Dockerfile.api` — CPU-only PyTorch image used by the ECS task.

## Infrastructure

- `infra/launch_training.py` — launches training jobs on **SageMaker** (`ml.g4dn.xlarge`) using a custom training image pushed to ECR. Reads W&B credentials from the environment, points the job at the cleaned splits in S3, and writes the final model to S3.
- `docker/Dockerfile.train` — training image used by SageMaker.
- `docker/Dockerfile.api` — inference image used by ECS.

## Repository layout

```
src/healthcare_nlp/        library code (data, model, train, api)
scripts/                   CLI entry points (prepare, train, evaluate)
experiments/               shell scripts that drive sweep stages
infra/                     SageMaker launcher
docker/                    training and inference Dockerfiles
tests/                     unit tests for preprocessing, data, model, train, baseline
notebooks/                 EDA notebook
```

## Running locally

```bash
# set up env
pip install -e .[dev]

# build the cleaned splits
python scripts/prepare_dataset.py

# run the baseline
python scripts/run_baseline.py

# fine-tune BERT (stage 1 example)
python scripts/run_training.py \
  --run-name stage1-lr1e-3 --group stage1 \
  --training-mode head_only --lr 1e-3 \
  --use-class-weights --epochs 5

# evaluate on the test set
python scripts/evaluate.py

# run the API locally
uvicorn healthcare_nlp.api.app:app --reload
```

## In progress / next steps

- **CI/CD pipeline.** I want to wire up a GitHub Actions workflow that runs the unit tests on every push, then on merges to `main` builds the training and inference Docker images, tags them (by git SHA and `latest`), and pushes them to the project's ECR repositories so the ECS service and SageMaker training image are always in sync with `main`. This closes the loop between "code merged" and "image deployable" without any manual `docker build`/`docker push` steps.
- **LoRA fine-tuning.** I'm planning to rerun the staged fine-tuning with **LoRA** (low-rank adapters) instead of the standard `top_n` / `full_training` schemes, to compare parameter-efficiency, final macro-F1, and train-time cost against the current baseline. The interesting question is whether LoRA at a very small rank can match the best `top_n` run while training <1% of the parameters — this would directly translate into cheaper SageMaker jobs and smaller artifacts to ship.
- **Inference drift monitoring.** I'm currently building drift monitoring on top of the API's JSON access logs: tracking input-distribution drift (token length, vocabulary overlap against the training set) and prediction-distribution drift (class ratio, mean confidence) over rolling windows, with alerts when a window diverges from the training-time reference. The goal is to make the service *observable as an ML system* and not just as a web service.
