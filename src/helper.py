import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


sys.path.append("./src/")

from utils import config


def dataset_initialization():

    X_train = pd.read_csv(
        os.path.join(config()["path"]["processed_path"], "X_train.csv")
    )
    X_test = pd.read_csv(
        os.path.join(config()["path"]["processed_path"], "X_test.csv"),
    )
    y_train = pd.read_csv(
        os.path.join(config()["path"]["processed_path"], "y_train.csv")
    )
    y_test = pd.read_csv(
        os.path.join(config()["path"]["processed_path"], "y_test.csv"),
    )

    training_dataset = pd.concat(
        [X_train, y_train],
        axis=1,
    )
    testing_dataset = pd.concat(
        [X_test, y_test],
        axis=1,
    )

    training_dataset = training_dataset.iloc[:, 4:]
    testing_dataset = testing_dataset.iloc[:, 4:]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "training_dataset": training_dataset,
        "testing_dataset": testing_dataset,
    }


def features_extraction_techquiue():
    dataset = dataset_initialization()

    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    training_dataset = dataset["training_dataset"]
    testing_dataset = dataset["testing_dataset"]

    decompositoon = PCA()
    decompositoon.fit_transform(training_dataset)
    print(np.cumsum(decompositoon.explained_variance_ratio_))

    best_n_components = 2

    decompositoon = PCA(n_components=2)
    X_transformed = decompositoon.fit_transform(training_dataset)
    y_transformed = decompositoon.transform(testing_dataset)

    X = pd.concat(
        [X_transformed, pd.concat([X_train, y_train], axis=1)["labels"]],
        axis=1,
    )
    y = pd.concat(
        [y_transformed, pd.concat([X_train, y_train], axis=1)["labels"]],
        axis=1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config()["dataloader"]["split_size"], random_state=42
    )

    os.makedirs(config()["path"]["processed_path"], "PCA-dataset")

    for type, dataset in [
        ("X_train", X_train),
        ("X_test", X_test),
        ("y_train", y_train),
        ("y_test", y_test),
    ]:
        dataset.to_csv(config()["path"]["processed_path"] + f"/{type}.csv", index=False)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


if __name__ == "__main__":
    print("Hello World")
