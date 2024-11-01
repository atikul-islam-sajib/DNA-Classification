import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


sys.path.append("./src/")

from utils import config
from helper import features_extraction_technique, features_selection_technique


class Trainer:
    def __init__(
        self,
        model: str = "RF",
        features_extraction: bool = False,
        features_selection: bool = False,
    ):
        self.steps = "training"
        self.features_extraction = features_extraction
        self.features_selection = features_selection

    def choose_dataset(self):
        if self.features_extraction:
            return features_extraction_technique()
        elif self.features_selection:
            return features_selection_technique()
        else:
            path = config()["path"]["processed_data"]
            return (
                {
                    "X_train": os.path.join(path, "X_train.csv"),
                    "X_test": os.path.join(path, "X_test.csv"),
                    "y_train": os.path.join(path, "y_train.csv"),
                    "y_test": os.path.join(path, "y_test.csv"),
                }
                if os.path.exists(config()["path"]["processed_data"])
                else "Make sure the processed data is in the right path".capitalize()
            )

    def train(self):
        dataset = self.choose_dataset()

        print(dataset["X_train"].shape)
        print(dataset["y_train"].shape)
        print(dataset["X_test"].shape)
        print(dataset["y_test"].shape)


if __name__ == "__main__":
    trainer = Trainer(features_selection=True)
    trainer.train()
