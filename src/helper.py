import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
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

    training_dataset = training_dataset.iloc[:, 2:]
    testing_dataset = testing_dataset.iloc[:, 2:]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "training_dataset": training_dataset,
        "testing_dataset": testing_dataset,
    }


def features_extraction_technique():
    dataset = dataset_initialization()
    training_dataset = dataset["training_dataset"]
    testing_dataset = dataset["testing_dataset"]

    pca = PCA()
    pca.fit(training_dataset)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    best_n_components = np.argmax(explained_variance >= 0.90) + 1
    print(f"Best number of components: {best_n_components}")

    pca = PCA(n_components=best_n_components)
    X_train_transformed = pca.fit_transform(training_dataset)
    X_test_transformed = pca.transform(testing_dataset)

    X_train_df = pd.DataFrame(
        X_train_transformed,
        index=dataset["X_train"].index,
    )
    X_test_df = pd.DataFrame(
        X_test_transformed,
        index=dataset["X_test"].index,
    )

    y_train = dataset["y_train"].reset_index(drop=True)["labels"]
    y_test = dataset["y_test"].reset_index(drop=True)["labels"]

    dataset = pd.concat(
        [
            pd.concat([X_train_df, y_train], axis=1),
            pd.concat([X_test_df, y_test], axis=1),
        ],
        axis=0,
    )

    X = dataset.drop("labels", axis=1)
    y = dataset["labels"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config()["dataloader"]["split_size"],
        random_state=42,
    )

    os.makedirs(
        os.path.join(config()["path"]["processed_path"], "PCA-dataset"), exist_ok=True
    )
    for dataset_name, data in [
        ("X_train", X_train),
        ("X_test", X_test),
        ("y_train", y_train),
        ("y_test", y_test),
    ]:
        data.to_csv(
            os.path.join(
                config()["path"]["processed_path"], "PCA-dataset", f"{dataset_name}.csv"
            ),
            index=False,
        )

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


if __name__ == "__main__":
    features_extraction_technique()
