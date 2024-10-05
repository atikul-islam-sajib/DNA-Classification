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

        self.dataset = pd.read_csv("./data/raw/DNA-Classification.csv")

    def feature_generator(self):
        if "single" in self.approaches:
            for instance in tqdm(range(self.dataset.shape[0])):
                for nucleoside in single_nucleosides:
                    for sequence in self.dataset.loc[instance, "sequence"]:
                        if nucleoside == sequence:
                            self.dataset[str(instance) + "_single_nucleoside"] = 1
                        else:
                            self.dataset[str(instance) + "_single_nucleoside"] = 0

        if "di" in self.approaches:
            for instance in tqdm(range(self.dataset.shape[0])):
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
            for instance in tqdm(range(self.dataset.shape[0])):
                for nucleoside in tri_nucleosides:
                    for index in range(len(self.dataset.loc[instance, "sequence"]) - 2):
                        if (
                            nucleoside
                            == self.dataset.loc[instance, "sequence"][index : index + 3]
                        ):
                            self.dataset[str(instance) + "_tri_nucleoside"] = 1
                        else:
                            self.dataset[str(instance) + "_tri_nucleoside"] = 0

        if "tetra" in self.approaches:
            for instance in tqdm(range(self.dataset.shape[0])):
                for nucleoside in tetra_nucleosides:
                    for index in range(len(self.dataset.loc[instance, "sequence"]) - 3):
                        if (
                            nucleoside
                            == self.dataset.loc[instance, "sequence"][index : index + 4]
                        ):
                            self.dataset[str(instance) + "_tetra_nucleoside"] = 1
                        else:
                            self.dataset[str(instance) + "_tetra_nucleoside"] = 0

        if "gc-content" in self.approaches:
            for instance in tqdm(range(self.dataset.shape[0])):
                sequence = self.dataset.loc[instance, "sequence"]

                A = sequence.count("A")
                C = sequence.count("C")
                G = sequence.count("G")
                T = sequence.count("T")

                GC_Content = (G + C) / (A + C + G + T)

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
