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


class Trainer:
    def __init__(
        self, features_extraction: bool = False, features_selection: bool = False
    ):
        self.steps = "training"
        self.features_extraction = features_extraction
        self.features_selection = features_selection

        self.number_of_components = 20

        self.X_train = os.path.join(config()["path"]["processed_path"], "X_train.csv")
        self.X_test = os.path.join(config()["path"]["processed_path"], "X_test.csv")
        self.y_train = os.path.join(config()["path"]["processed_path"], "y_train.csv")
        self.y_test = os.path.join(config()["path"]["processed_path"], "y_test.csv")

        self.X_train = pd.read_csv(self.X_train)
        self.X_test = pd.read_csv(self.X_test)
        self.y_train = pd.read_csv(self.y_train)
        self.y_test = pd.read_csv(self.y_test)

        self.training_dataset = pd.concat([self.X_train, self.y_train], axis=1)
        self.testing_dataset = pd.concat([self.X_test, self.y_test], axis=1)

        self.training_dataset = self.training_dataset.iloc[:, 4:]
        self.testing_dataset = self.testing_dataset.iloc[:, 4:]

        print(self.training_dataset.head(2))
        print(self.training_dataset.isnull().sum().sum())

        print(self.training_dataset.iloc[2, :].values)

    def feature_extraction_technique(self):
        self.explained_variances = []

        print(self.training_dataset.shape, self.training_dataset.isnull().sum().sum())
        print(self.training_dataset.head())

        decompositoon = PCA()
        decompositoon.fit_transform(self.training_dataset)
        print(decompositoon.explained_variance_ratio_)

    def train(self):
        pass


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

    # trainer.feature_extraction_technique()
