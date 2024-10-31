import sys
import argparse
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.append("./src/")

from utils import config


class MachineLearningModel:
    def __init__(self, model: str = "RF"):
        self.model = model

    def define_model(self):
        if self.model == "RF":
            return RandomForestClassifier()
        elif self.model == "DT":
            return DecisionTreeClassifier()
        elif self.model == "NB":
            return MultinomialNB()
        elif self.model == "LR":
            return LogisticRegression()
        elif self.model == "XGB":
            return XGBClassifier()
        else:
            raise TypeError(
                "Select the appropriate machine learning model to train the model".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the model for the DNA-Classification".title()
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config()["model"]["model_name"],
        choices=["RF", "DT", "NB", "LR", "XGB"],
        help="Define the model name for the model".capitalize(),
    )
    
    model = MachineLearningModel()
    model.define_model()
