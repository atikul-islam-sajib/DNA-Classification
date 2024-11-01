import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


sys.path.append("./src/")

from utils import config
from helper import features_extraction_techquiue


class Trainer:
    def __init__(
        self,
        model: str = "RF",
        features_extraction: bool = False,
        features_selection: bool = False,
    ):
        self.steps = "training"
        self.features_extraction = features_extraction
        self.features_selection = features_selection

    def train(self):
        if self.features_extraction:
            dataset = features_extraction_techquiue()
        elif self.features_selection:
            pass


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

    # trainer.feature_extraction_technique()
