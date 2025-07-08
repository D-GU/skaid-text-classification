import os

from torch.utils.data import Dataset, TensorDataset

from ..document_handler.text_preprocessor import TextPreprocessor


class DatasetNews(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.preprocessor = TextPreprocessor()

    @staticmethod
    def __read(filepath: str):
        with open(filepath) as text:
            text = text.read()
        return text

    def __getitem__(self, index):
        filepath = os.listdir(self.path)[index]
        text = self.__read(os.path.join(self.path, filepath))
        text = self.preprocessor.preprocess()

if __name__ == "__main__":
    ds = DatasetNews("../../20News")
