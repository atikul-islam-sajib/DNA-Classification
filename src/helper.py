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


def dataset_initialization():

    X_train = os.path.join(config()["path"]["processed_path"], "X_train.csv")
    X_test = os.path.join(config()["path"]["processed_path"], "X_test.csv")
    y_train = os.path.join(config()["path"]["processed_path"], "y_train.csv")
    y_test = os.path.join(config()["path"]["processed_path"], "y_test.csv")

    X_train = pd.read_csv(X_train)
    X_test = pd.read_csv(X_test)
    y_train = pd.read_csv(y_train)
    y_test = pd.read_csv(y_test)

    training_dataset = pd.concat([X_train, y_train], axis=1)
    testing_dataset = pd.concat([X_test, y_test], axis=1)

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
    print(decompositoon.explained_variance_ratio_)