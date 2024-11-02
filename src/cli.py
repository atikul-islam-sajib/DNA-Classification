import os
import sys
import argparse

sys.path.append("./src/")

from utils import config
from feature_generator import FeatureGenerator
from dataloader import DataLoader
from trainer import Trainer


def cli():
    parser = argparse.ArgumentParser(
        description="CLI command to train the DNA-Classification task".title()
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=config()["dataloader"]["dataset"],
        help="The dataset to be used for the task".capitalize(),
    )

    parser.add_argument(
        "--approaches",
        type=str,
        nargs="+",
        default=config()["dataloader"]["approaches"],
        help="The approaches to be used for the task".capitalize(),
    )

    parser.add_argument(
        "--split_size",
        type=float,
        default=config()["dataloader"]["split_size"],
        help="The split size to be used for the task".capitalize(),
    )

    parser.add_argument(
        "--FE",
        type=bool,
        default=config()["trainer"]["features_extraction"],
        choices=[True, False],
        help="Features Extraction".capitalize(),
    )
    parser.add_argument(
        "--FS",
        type=bool,
        default=config()["trainer"]["features_selection"],
        choices=[True, False],
        help="Feature Selection".capitalize(),
    )
    parser.add_argument(
        "--HP",
        type=bool,
        default=config()["trainer"]["hyperparameter_tuning"],
        choices=[True, False],
        help="Hyperparameter Tuning".capitalize(),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RF",
        choices=["RF", "DT", "LR", "XGB", "NB"],
        help="Model".capitalize(),
    )

    parser.add_argument(
        "--KFold",
        type=int,
        default=config()["trainer"]["KFold"],
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="K-Fold".capitalize(),
    )

    args = parser.parse_args()

    features_extraction = args.FE
    feature_selection = args.FS

    try:
        dataloader = DataLoader(
            dataset=args.dataset, approaches=args.approaches, split_size=args.split_size
        )
    except Exception as e:
        print(f"Error initializing DataLoader: {e}")
        exit(1)

    try:
        dataloader.feature_generator()
    except Exception as e:
        print(f"Error in feature generation: {e}")
        exit(1)

    try:
        _ = dataloader.split_dataset()
    except Exception as e:
        print(f"Error in splitting the dataset: {e}")
        exit(1)

    try:
        dataloader.dataset_history()
    except Exception as e:
        print(f"Error in recording dataset history: {e}")
        exit(1)

    trainer = Trainer(
        features_extraction=features_extraction,
        features_selection=feature_selection,
        hyperparameter_tuning=args.HP,
        model=args.model,
        KFold=args.KFold,
    )

    trainer.train()


if __name__ == "__main__":
    cli()
