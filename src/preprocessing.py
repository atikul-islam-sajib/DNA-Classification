import sys
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("./src")

from utils import single_nucleosides, di_nucleosides, tri_nucleosides, tetra_nucleosides


class Preprocessing:
    def __init__(self, approaches: list = ["single", "di", "tri", "tetra"]):
        self.approaches = approaches

        self.X = list()
        self.y = list()

        self.dataset = pd.read_csv("./data/raw/DNA-Classification.csv")

    def feature_generator(self):
        if "single" in self.approaches:
            for instance in tqdm(range(self.dataset[0:100].shape[0])):
                for nucleoside in single_nucleosides:
                    for sequence in self.dataset.loc[instance, "sequence"]:
                        if nucleoside == sequence:
                            self.dataset[str(instance) + "_single_nucleoside"] = 1
                        else:
                            self.dataset[str(instance) + "_single_nucleoside"] = 0

        if "di" in self.approaches:
            for instance in tqdm(range(self.dataset[0:100].shape[0])):
                for nucleoside in di_nucleosides:
                    for index in range(len(self.dataset.loc[instance, "sequence"]) - 1):
                        if (
                            nucleoside
                            == self.dataset.loc[instance, "sequence"][index : index + 2]
                        ):
                            self.dataset[str(instance) + "_di_nucleoside"] = 1
                        else:
                            self.dataset[str(instance) + "_di_nucleoside"] = 0

        if "tri" in self.approaches:
            for instance in tqdm(range(self.dataset[0:100].shape[0])):
                for nucleoside in tri_nucleosides:
                    for index in range(len(self.dataset.loc[instance, "sequence"]) - 2):
                        if (
                            nucleoside
                            == self.dataset.loc[instance, "sequence"][index : index + 3]
                        ):
                            self.dataset[str(instance) + "_tri_nucleoside"] = 1
                        else:
                            self.dataset[str(instance) + "_tri_nucleoside"] = 0

            print(self.dataset.isnull().sum().sum())
            print(self.dataset.shape)
            print(self.dataset.head())

        if "tetra" in self.approaches:
            for instance in tqdm(range(self.dataset[0:100].shape[0])):
                for nucleoside in tetra_nucleosides:
                    for index in range(len(self.dataset.loc[instance, "sequence"]) - 3):
                        if (
                            nucleoside
                            == self.dataset.loc[instance, "sequence"][index : index + 4]
                        ):
                            self.dataset[str(instance) + "_tri_nucleoside"] = 1
                        else:
                            self.dataset[str(instance) + "_tri_nucleoside"] = 0

            print(self.dataset.isnull().sum().sum())
            print(self.dataset.shape)
            print(self.dataset.head())


if __name__ == "__main__":
    preprocess = Preprocessing()

    preprocess.feature_generator()
