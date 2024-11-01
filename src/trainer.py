import os
import sys
import math
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    classification_report,
)


sys.path.append("./src/")

from utils import config, hyperparameter_tuning
from helper import features_extraction_technique, features_selection_technique
from model import MachineLearningModel

import warnings

warnings.filterwarnings("ignore")


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

    def model_evaluation(self, **kwargs):
        with open("./evaluation.json", "w") as file:
            json.dump(
                {
                    "Accuracy": np.mean(kwargs["accuracy"]).round(2),
                    "Precision": np.mean(kwargs["precision"]).round(2),
                    "Recall": np.mean(kwargs["recall"]).round(2),
                    "F1 Score": np.mean(kwargs["f1_score"]).round(2),
                    "Classification Report": classification_report(
                        kwargs["actual_labels"],
                        kwargs["predicted_labels"],
                        output_dict=True,
                    ),
                },
                file,
                indent=4,
            )

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

            predicted = classifier.predict(dataset["X_test"])

            print("The best parameters are: ".capitalize(), classifier.best_params_)
            print("Refined best parameters: ".capitalize(), classifier.best_score_)

            self.model_evaluation(
                accuracy=accuracy_score(predicted, dataset["y_test"]),
                precision=precision_score(
                    predicted, dataset["y_test"], average="weighted"
                ),
                recall=recall_score(predicted, dataset["y_test"], average="weighted"),
                f1_score=f1_score(predicted, dataset["y_test"], average="weighted"),
                actual_labels=dataset["y_test"],
                predicted_labels=predicted,
            )

        else:
            predicted_labels = []
            actual_labels = []

            KFoldCV = KFold(n_splits=self.KFold, shuffle=True, random_state=42)

            for index, (train_index, test_index) in enumerate(
                KFoldCV.split(dataset["X_train"], dataset["y_train"])
            ):
                print(
                    "*" * 10,
                    "KFold CV - {} is executing".format(index + 1).title(),
                    "*" * 10,
                )
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

                predicted_labels.extend(predicted)
                actual_labels.extend(y_test_fold)

                self.model_evaluation(
                    accuracy=self.accuracy,
                    precision=self.precision,
                    recall=self.recall,
                    f1_score=self.f1_score,
                    predicted_labels=predicted_labels,
                    actual_labels=actual_labels,
                )

            print(
                "The evaluation metrics are saved in the evaluation.json file".capitalize()
            )


if __name__ == "__main__":
    trainer = Trainer(
        features_extraction=True, hyperparameter_tuning=True, model="RF", KFold=5
    )
    trainer.train()
