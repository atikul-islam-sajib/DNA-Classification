from sklearn.ensemble import RandomForestClassifier


class MachineLearningModel:
    def __init__(self, model: str = "RF"):
        self.model = model
        
    def define_model(self):
        pass
    
    
if __name__ == "__main__":
    model = MachineLearningModel()
    model.define_model()