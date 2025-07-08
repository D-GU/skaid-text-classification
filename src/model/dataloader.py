import os
import json
import glob

from torch.utils.data import Dataset

from src.document_handler.text_preprocessor import TextPreprocessor

CATEGORIES = json.load(open("categories.json"))


class DatasetNews(Dataset):
    def __init__(self, path: str):
        self.text_paths = [txt_path for txt_path in sorted(glob.glob(f"{path}/*"))]

        self.preprocessor = TextPreprocessor()
        self.categories = CATEGORIES

        self.cls_names = {}

        for idx, txt_path in enumerate(self.text_paths):
            cls_name = self.__get_class_name(txt_path)
            self.cls_names[cls_name] = idx

    @staticmethod
    def __get_class_name(path):
        return os.path.basename(path)[0:-4]

    @staticmethod
    def __read(filepath: str):
        text = None
        with open(filepath, encoding="windows-1252") as file:
            text = file.read()
        return text

    def __len__(self):
        return len(self.text_paths)

    def __getitem__(self, index):
        # Считаем текст
        text_path = self.text_paths[index]
        text = self.__read(text_path)

        # Векторизируем текст
        x = self.preprocessor(text)
        cls_name = self.__get_class_name(text_path)
        target = self.cls_names[cls_name]

        return x, target


if __name__ == "__main__":
    ds = DatasetNews("../../20News")
