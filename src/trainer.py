import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


sys.path.append("./src/")

from utils import config, hyperparameter_tuning
from helper import features_extraction_technique, features_selection_technique
from model import MachineLearningModel


class Trainer:
    def __init__(
        self,
        model: str = "RF",
        features_extraction: bool = False,
        features_selection: bool = False,
        hyperparameter_tuning: bool = False,
        KFold: int = 5,
    ):
        self.steps = "training"
        self.model = model
        self.features_extraction = features_extraction
        self.features_selection = features_selection
        self.hyperparameter_tuning = hyperparameter_tuning
        self.KFold = KFold

        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1_score = []

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

        if self.hyperparameter_tuning:
            classifier = GridSearchCV(
                estimator=classifier,
                param_grid=hyperparameter_tuning(model=self.model),
                cv=self.KFold,
                scoring="accuracy",
            )

            classifier.fit(dataset["X_train"], dataset["y_train"])
            print("The best parameters are: ".capitalize(), classifier.best_params_)
            print("Refined best parameters: ".capitalize(), classifier.best_score_)

        else:
            KFoldCV = KFold(n_splits=self.KFold, shuffle=True, random_state=42)

            for train_index, test_index in KFoldCV.split(
                dataset["X_train"], dataset["y_train"]
            ):
                X_train_fold, y_train_fold = (
                    dataset["X_train"].iloc[train_index, :],
                    dataset["y_train"].iloc[train_index],
                )
                X_test_fold, y_test_fold = (
                    dataset["X_train"].iloc[test_index, :],
                    dataset["y_train"].iloc[test_index],
                )

                classifier.fit(X_train_fold, y_train_fold)

                predicted = classifier.predict(X_test_fold)

                self.accuracy.append(accuracy_score(y_test_fold, predicted))
                self.precision.append(
                    precision_score(y_test_fold, predicted, average="weighted")
                )
                self.recall.append(
                    recall_score(y_test_fold, predicted, average="weighted")
                )
                self.f1_score.append(
                    f1_score(y_test_fold, predicted, average="weighted")
                )

            print("Average Accuracy: ", np.mean(self.accuracy))
            print("Average Precision: ", np.mean(self.precision))
            print("Average Recall: ", np.mean(self.recall))
            print("Average F1 Score: ", np.mean(self.f1_score))


if __name__ == "__main__":
    trainer = Trainer(
        features_extraction=True, hyperparameter_tuning=False, model="RF", KFold=5
    )
    trainer.train()
