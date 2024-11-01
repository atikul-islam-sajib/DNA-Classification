import sys
import argparse


sys.path.append("./src/")


class Trainer:
    def __init__(
        self, features_extraction: bool = False, features_selection: bool = False
    ):
        self.steps = "training"
        self.features_extraction = features_extraction
        self.features_selection = features_selection

    def feature_extraction_technique(self):
        pass

    def train(self):
        pass


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
