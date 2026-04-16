from __future__ import annotations
import argparse
import os
from dotenv import load_dotenv
import boto3
from sagemaker.modules.train import ModelTrainer  # type: ignore
from sagemaker.modules.configs import InputData, Compute, OutputDataConfig  # type: ignore

load_dotenv()

ROLE = os.environ["ROLE"]
IMAGE = os.environ["IMAGE"]
S3_DATA = os.environ["S3_DATA"]
S3_OUTPUT = os.environ["S3_OUTPUT"]
boto3.setup_default_session(region_name="us-east-2")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch training jobs in SageMaker."
    )
    parser.add_argument(
        "--group",
        type=str,
        required=True,
        help="Group or stage of experiments."
    )
    parser.add_argument(
        "--lr",
        required=True,
        type=float,
        help="Learning rate."
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        default="top_n",
        choices=["head_only", "full_training", "top_n"],
        help="Which layers to train: 'head_only' (classifier only), 'full_training' (all), or 'top_n' (last n encoder layers + head).",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=6,
        help="This is the number of layers to be unfrozen if training mode is top_n."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="NUmber of epochs."
    )
    args = parser.parse_args()

    hyperparameters = {
        "run-name": args.group + "-" + args.training_mode + "-" + str(args.n_layers) + "-" + str(args.lr),
        "group": args.group,
        "lr": args.lr,
        "training-mode": args.training_mode,
        "n-layers": args.n_layers,
        "epochs": args.epochs,
        "path-to-splits": "/opt/ml/input/data/training/",
        "output-dir": "/opt/ml/model/",
        "checkpoint-path": "/opt/ml/model/best.pt",
    }

    trainer = ModelTrainer(
        training_image=IMAGE,
        role=ROLE,
        compute=Compute(instance_type="ml.g4dn.xlarge", instance_count=1),
        environment={"WANDB_API_KEY": os.environ["WANDB_API_KEY"]},
        output_data_config=OutputDataConfig(
            s3_output_path=f"{S3_OUTPUT}{hyperparameters['run-name']}/"
        ),
        hyperparameters=hyperparameters,
    )

    train_data = InputData(channel_name="training", data_source=S3_DATA)
    trainer.train(input_data_config=[train_data])

if __name__ == "__main__":
    main()
