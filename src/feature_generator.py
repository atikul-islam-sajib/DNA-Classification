import os
import sys
import argparse
import warnings
import traceback
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.append("./src")

from utils import (
    single_nucleosides,
    di_nucleosides,
    tri_nucleosides,
    tetra_nucleosides,
    config,
)


class FeatureGenerator:
    def __init__(
        self, approaches: list = ["single", "di", "tri", "tetra", "gc-content"]
    ):
        self.approaches = approaches

        self.X = list()
        self.y = list()

        self.GC_Content = list()

        self.dataset = pd.read_csv("./data/raw/DNA-Classification.csv")[0:5]

    def feature_generator(self):
        if "single" in self.approaches:
            max_len = max(self.dataset["sequence"].apply(len))

            for pos in range(max_len):
                for nucleoside in single_nucleosides:
                    feature_column = f"{nucleoside}_pos_{pos}"
                    self.dataset[feature_column] = 0

            for instance in tqdm(range(self.dataset.shape[0])):
                sequence = self.dataset.loc[instance, "sequence"]

                for pos, nucleotide in enumerate(sequence):
                    for nucleoside in single_nucleosides:
                        feature_column = f"{nucleoside}_pos_{pos}"
                        if nucleoside == nucleotide:
                            self.dataset.loc[instance, feature_column] = 1

        if "di" in self.approaches:
            max_len = max(self.dataset["sequence"].apply(len))

            for pos in range(max_len - 1):
                for di_nucleoside in di_nucleosides:
                    feature_column = f"{di_nucleoside}_pos_{pos}_di_nucleoside"
                    self.dataset[feature_column] = 0

            for instance in tqdm(range(self.dataset.shape[0])):
                sequence = self.dataset.loc[instance, "sequence"]
                for pos in range(len(sequence) - 1):
                    for di_nucleoside in di_nucleosides:
                        feature_column = f"{di_nucleoside}_pos_{pos}_di_nucleoside"
                        if sequence[pos : pos + 2] == di_nucleoside:
                            self.dataset.loc[instance, feature_column] = 1

        if "tri" in self.approaches:
            max_len = max(self.dataset["sequence"].apply(len))

            for pos in range(max_len - 2):
                for tri_nucleoside in tri_nucleosides:
                    feature_column = f"{tri_nucleoside}_pos_{pos}_tri_nucleoside"
                    self.dataset[feature_column] = 0

            for instance in tqdm(range(self.dataset.shape[0])):
                sequence = self.dataset.loc[instance, "sequence"]
                for pos in range(len(sequence) - 2):
                    for tri_nucleoside in tri_nucleosides:
                        feature_column = f"{tri_nucleoside}_pos_{pos}_tri_nucleoside"
                        if sequence[pos : pos + 3] == tri_nucleoside:
                            self.dataset.loc[instance, feature_column] = 1

        if "tetra" in self.approaches:
            max_len = max(self.dataset["sequence"].apply(len))

            for pos in range(max_len - 3):
                for tetra_nucleoside in tetra_nucleosides:
                    feature_column = f"{tetra_nucleoside}_pos_{pos}_tetra_nucleoside"
                    self.dataset[feature_column] = 0

            for instance in tqdm(range(self.dataset.shape[0])):
                sequence = self.dataset.loc[instance, "sequence"]
                for pos in range(len(sequence) - 3):
                    for tetra_nucleoside in tetra_nucleosides:
                        feature_column = (
                            f"{tetra_nucleoside}_pos_{pos}_tetra_nucleoside"
                        )
                        if sequence[pos : pos + 4] == tetra_nucleoside:
                            self.dataset.loc[instance, feature_column] = 1

        if "gc-content" in self.approaches:
            self.GC_Content = []

            for instance in tqdm(range(self.dataset.shape[0])):
                sequence = self.dataset.loc[instance, "sequence"]
                G_count = sequence.count("G")
                C_count = sequence.count("C")
                GC_Content = (
                    (G_count + C_count) / len(sequence) if len(sequence) > 0 else 0
                )
                self.GC_Content.append(GC_Content)

            self.dataset["GC-Content"] = self.GC_Content

        try:
            self.dataset.to_csv(
                os.path.join(
                    config()["path"]["processed_path"], "processed_dataset.csv"
                )
            )
        except Exception as e:
            print(
                "Cannot saved the dataset in the processed file, & error: {}".capitalize().format(
                    e
                )
            )
            traceback.print_exc()
        else:
            print(
                "the dataset stored in the {} folder".format(
                    config()["path"]["processed_path"]
                ).capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature Generator for DNA-Classification".title()
    )

    parser.add_argument(
        "--single",
        action="store_true",
        help="Feature Generator using approach - single".capitalize(),
    )
    parser.add_argument(
        "--di",
        action="store_true",
        help="Feature Generator using approach - di".capitalize(),
    )
    parser.add_argument(
        "--tri",
        action="store_true",
        help="Feature Generator using approach - tri".capitalize(),
    )
    parser.add_argument(
        "--tetra",
        action="store_true",
        help="Feature Generator using approach - tetra".capitalize(),
    )
    parser.add_argument(
        "--gc_content",
        action="store_true",
        help="Feature Generator using approach - gc-content".capitalize(),
    )

    args = parser.parse_args()

    approaches = [
        "single" if args.single else False,
        "di" if args.di else False,
        "tri" if args.tri else False,
        "tetra" if args.tetra else False,
        "gc-content" if args.gc_content else False,
    ]

    feature_generator = FeatureGenerator(approaches=approaches)

    feature_generator.feature_generator()
