import sys


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
        pass
        
    def load_dataset(self):
        pass
    
    @staticmethod
    def dataset_history():
        pass
    
if __name__ == "__main__":
    dataloader = DataLoader(dataset=None, split_size=0.20)
        
