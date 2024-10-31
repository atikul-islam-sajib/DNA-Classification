import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("./src")

from utils import config
from feature_generator import FeatureGenerator


class DataLoader:
    def __init__(
        self,
        dataset=None,
        split_size: float = 0.20,
        approaches: list = ["single", "di", "tri", "tetra", "gc_content"],
    ):
        self.dataset = dataset
        self.split_size = split_size
        self.approaches = approaches

    def split_dataset(self):
        if os.path.exists(config()["path"]["processed_path"]):
            dataset = os.path.join(
                config()["path"]["processed_path"], "processed_dataset.csv"
            )

            self.processed_data = pd.read_csv(dataset)

            X = self.processed_data.iloc[:, :-1]
            y = self.processed_data.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, split_size=self.split_size, random_state=42
            )

            for type, dataset in [
                ("X_train", X_train),
                ("X_test", X_test),
                ("y_train", y_train),
                ("y_test", y_test),
            ]:
                dataset.to_csv(
                    os.path.join(config()["path"]["processed_path"], f"{type}.csv"),
                    index=False,
                )

            print(
                "Training and testing dataset is stored in the folder {}".format(
                    config()["path"]["processed_path"]
                )
            )

            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }

    def feature_generator(self):
        if isinstance(self.approaches, list):
            self.generator = FeatureGenerator(approaches=self.approaches)

            try:
                self.generator.feature_generator()
            except Exception as e:
                print(f"An error occurred: {e}".capitalize())
            else:
                print(
                    "Feature generation completed successfully and store in the folder {}".format(
                        os.path.join(config()["path"]["processed_path"])
                    )
                )
        else:
            raise ValueError("Approaches must be a list".capitalize())

    @staticmethod
    def dataset_history():
        if os.path.exists(config()["path"]["processed_path"]):
            processed_path = os.path.join(
                config()["path"]["processed_path"], "processed_dataset.csv"
            )

            dataset = pd.read_csv(processed_path)

            information = {}

            information["isNaN".title()] = (
                "NaN".capitalize()
                if dataset.isnull().sum().sum() > 0
                else "no NaN".capitalize()
            )

            information["total_features".title()] = str(dataset.shape[1])
            information["total_instances".title()] = str(dataset.shape[0])
            information["dataset_shape".title()] = str(dataset.shape)
            information["target_ratio".title()] = str(
                dataset["labels"].value_counts(ascending=False).to_dict()
            )

            pd.DataFrame(information, index=[0]).to_csv(
                os.path.join(config()["path"]["files_path"], "dataset_history.csv")
            )

            print(
                "Dataset history is stored in the folder {}".format(
                    config()["path"]["files_path"]
                )
            )


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DataLoader and feature generator for the DNA-Classification task".title()
    )

    parser.add_argument(
        "--dataset",
        type=config()["dataloader"]["dataset"],
        default=None,
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

    args = parser.parse_args()

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
        splits_dataset = dataloader.split_dataset()
    except Exception as e:
        print(f"Error in splitting the dataset: {e}")
        exit(1)

    try:
        dataloader.dataset_history()
    except Exception as e:
        print(f"Error in recording dataset history: {e}")
        exit(1)
