import os

from torch.utils.data import Dataset, TensorDataset

from src.document_handler.text_preprocessor import TextPreprocessor


class DatasetNews(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.preprocessor = TextPreprocessor()

    @staticmethod
    def __read(filepath: str):
        text = None
        with open(filepath, encoding="windows-1252") as file:
            text = file.read()
        return text

    def __getitem__(self, index):
        filepath = os.listdir(self.path)[index]
        text = self.__read(os.path.join(self.path, filepath))
        text = self.preprocessor(text)
        print(text)


if __name__ == "__main__":
    ds = DatasetNews("../../20News")
    ds[0]
