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
from model import MachineLearningModel


class Trainer:
    def __init__(
        self,
        model: str = "RF",
        features_extraction: bool = False,
        features_selection: bool = False,
        KFold: int = 5,
    ):
        self.steps = "training"
        self.model = model
        self.features_extraction = features_extraction
        self.features_selection = features_selection
        self.KFold = KFold

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

    def select_the_model(self):
        if self.model == "RF":
            return MachineLearningModel(model="RF").define_model()
        elif self.model == "DT":
            return MachineLearningModel(model="DT").define_model()
        elif self.model == "LR":
            return MachineLearningModel(model="LR").define_model()
        elif self.model == "XGB":
            return MachineLearningModel(model="XGB").define_model()
        elif self.model == "NB":
            return MachineLearningModel(model="NB").define_model()
        else:
            return "Make sure the model is in the right format".capitalize()

    def train(self):
        dataset = self.choose_dataset()
        classifier = self.select_the_model()

        classifier.fit(dataset["X_train"], dataset["y_train"])

        print(classifier.score(dataset["X_test"], dataset["y_test"]))


if __name__ == "__main__":
    trainer = Trainer(features_selection=True)
    trainer.train()
