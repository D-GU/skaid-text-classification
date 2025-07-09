#!/usr/bin/env python3
import argparse
import json

import torch

from src.document_handler.parser import Parser
from src.document_handler.text_preprocessor import TextPreprocessor
from src.model.model import TextClassifier
from src.model.model_cfg import ModelCFG

# Список категорий в том же порядке, что и модель выдает
CATEGORIES = json.load(open("src/model/categories.json"))


def load_model(cfg: ModelCFG) -> TextClassifier:
    model = TextClassifier(cfg)
    model.load_state_dict(torch.load("document_classifier.pth", weights_only=True))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Классификация PDF-документа в одну из 20 тем.")
    parser.add_argument("pdf_path", help="Путь к PDF-файлу для классификации")
    args = parser.parse_args()

    # Загружаем текст
    txt = Parser()(args.pdf_path)

    # Очищаем и токенизируем
    tokens = TextPreprocessor()(txt)

    cfg = ModelCFG()

    # Загружаем модель
    model = load_model(cfg)

    vectorizer = model.train_dataloader().dataset.vectorizer
    to_predict = torch.tensor(vectorizer.transform(tokens)).unsqueeze(0)

    with torch.no_grad():
        logits = model(to_predict)
    pred = torch.argmax(logits, dim=1).item()

    # Выводим результат
    print(f"Предсказанная категория: {CATEGORIES[pred]}")


if __name__ == "__main__":
    main()
