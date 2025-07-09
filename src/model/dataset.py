import json
import pickle

import torch
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset, DataLoader

from src.document_handler.text_preprocessor import TextPreprocessor
from src.document_handler.vectorizer import Vectorizer

class NewsgroupsDataset(Dataset):
    def __init__(self, subset="train", vectorizer: Vectorizer = None):
        self.bundle = fetch_20newsgroups(subset=subset, shuffle=True, random_state=42)
        self.raw_texts = self.bundle.data
        self.targets = self.bundle.target

        self.cleaner = TextPreprocessor()
        self.vectorizer = vectorizer or Vectorizer()

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


if __name__ == "__main__":
    train_dataset = NewsgroupsDataset(subset="train")

    test_dataset = NewsgroupsDataset(
        subset="test",
        vectorizer=train_dataset.vectorizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,  # по желанию
        pin_memory=True  # по желанию для GPU
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: features {features.shape}, labels {labels.shape}")
        break
