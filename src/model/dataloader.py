import json

import torch
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset, DataLoader, DataLoader

from src.document_handler.text_preprocessor import TextPreprocessor
from src.document_handler.vectorizer import Vectorizer

CATEGORIES = json.load(open("categories.json"))


class NewsgroupsDataset(Dataset):
    def __init__(self, subset="train"):
        self.bundle = fetch_20newsgroups(subset=subset, shuffle=True, random_state=42)
        self.raw_texts = self.bundle.data
        self.targets = self.bundle.target

        self.cleaner = TextPreprocessor()
        self.vectorizer = Vectorizer()

        self.tokens_list = [self.cleaner(text) for text in self.raw_texts]

        # 2) строим единый словарь по всем документам (только на train)
        if subset == "train":
            self.vectorizer.fit(self.tokens_list)

        # 3) векторизуем
        self.vectors = [torch.tensor(self.vectorizer.transform(token)) for token in self.tokens_list]

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx], torch.tensor(self.targets[idx], dtype=torch.long)


def get_dataloader(
        subset: str, batch_size: int, shuffle: bool, num_workers: int
):
    dl = DataLoader(
        NewsgroupsDataset(subset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dl


if __name__ == "__main__":
    dl = get_dataloader("train", 32, False, 1)
    x, y = next(iter(dl))
    print(x)


