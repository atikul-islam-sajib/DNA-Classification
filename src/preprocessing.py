import sys
import numpy as np
import pandas as pd 
from tqdm import tqdm

sys.path.append("./src")

from utils import single_nucleosides


class Preprocessing():
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
                            self.dataset[str(instance)+"_single_nucleoside"] = 1
                        else:
                            self.dataset[str(instance)+"_single_nucleoside"] = 0
                            
            print(self.dataset.isnull().sum().sum())
            print(self.dataset.shape)
            print(self.dataset.head())
                            
                    
                    

if __name__ == "__main__":
    preprocess = Preprocessing()
    
    preprocess.feature_generator()