import os
import sys
import argparse
import traceback
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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
        os.path.join(config()["path"]["processed_path"], "y_train.csv"),
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

    training_dataset = training_dataset.iloc[:, :-1]
    testing_dataset = testing_dataset.iloc[:, :-1]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "training_dataset": training_dataset,
        "testing_dataset": testing_dataset,
    }


def features_extraction_technique():
    try:
        dataset = dataset_initialization()

        training_dataset = dataset["training_dataset"]
        testing_dataset = dataset["testing_dataset"]

    except Exception as e:
        print("An error is occured: ", e)

    try:
        pca = PCA()
        pca.fit(training_dataset)

    except ImportError as e:
        print("An error is occured: ", e)
    except Exception as e:
        print("An error is occured: ", e)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    best_n_components = np.argmax(explained_variance >= 0.90) + 1

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

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def features_selection_technique():
    RF = RandomForestClassifier(n_estimators=300, criterion="gini", random_state=42)

    try:
        dataset = dataset_initialization()

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_test = dataset["X_test"]
        y_test = dataset["y_test"]

        RF.fit(X_train, y_train)

        feature_importances = RF.feature_importances_

        importance_df = pd.concat(
            [
                pd.DataFrame(X_train.columns, columns=["Features"]),
                pd.DataFrame(feature_importances, columns=["Importance"]),
            ],
            axis=1,
        ).sort_values(by=["Importance"], ascending=False)

        columns = importance_df[importance_df["Importance"] >= 0.001]["Features"].values
        index = importance_df[importance_df["Importance"] >= 0.001].index

        X_train = X_train.loc[:, columns]
        X_test = X_test.loc[:, columns]

        os.makedirs(
            os.path.join(config()["path"]["processed_path"], "Feature-Importance"),
            exist_ok=True,
        )

        for dataset_name, data in [
            ("X_train", X_train),
            ("X_test", X_test),
            ("y_train", y_train),
            ("y_test", y_test),
        ]:
            data.to_csv(
                os.path.join(
                    config()["path"]["processed_path"],
                    "Feature-Importance",
                    f"{dataset_name}.csv",
                ),
                index=False,
            )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    except Exception as e:
        print("An error occurred: ", e)
        traceback.print_exc()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Helper method for the DNA-Classifier".title()
    # )
    # parser.add_argument(
    #     "--FE",
    #     type=str,
    #     default="PCA",
    #     help="Features Extraction Technique".capitalize(),
    # )
    # parser.add_argument(
    #     "--FS",
    #     type=bool,
    #     default=False,
    #     help="Features Selection Technique".capitalize(),
    # )

    # args = parser.parse_args()

    # if args.FE:
    #     _ = features_extraction_technique()
    # elif args.FS:
    #     _ = features_selection_technique()

    features_selection_technique()
