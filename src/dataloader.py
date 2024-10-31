from utils import config
from feature_generator import FeatureGenerator
import os
import sys
import argparse
from sklearn.model_selection import train_test_split

sys.path.append("./src")


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
            self.processed_data = os.path.join(config()["path"]["processed_path"], "processed_dataset.csv")
            
            X = self.processed_data.iloc[:, :-1].values
            y = self.processed_data.iloc[:, -1].values
            
                         
    def feature_generator(self):
        if isinstance(self.approaches, list):
            self.generator = FeatureGenerator(approaches=self.approaches)

            try:
                self.generator.feature_generator()
            except Exception as e:
                print(f"An error occurred: {e}".capitalize())
            else:
                print("Feature generation completed successfully and store in the folder {}".format(
                    os.path.join(config()["path"]["processed_path"])))
        else:
            raise ValueError("Approaches must be a list".capitalize())

    @staticmethod
    def dataset_history():
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DataLoader and feature generator for the DNA-Classification task".title())

    parser.add_argument("--dataset", type=str, default=None,
                        help="The dataset to be used for the task".capitalize())

    parser.add_argument("--approaches", type=str, nargs='+', default=["single", "di", "tri", "tetra", "gc_content"],
                        help="The approaches to be used for the task".capitalize())

    parser.add_argument("--split_size", type=float, default=0.20,
                        help="The split size to be used for the task".capitalize())

    args = parser.parse_args()

    dataloader = DataLoader(
        dataset=args.dataset, approaches=args.approaches, split_size=args.split_size)
