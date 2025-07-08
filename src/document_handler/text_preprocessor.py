import re

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from parser import Parser


# Необходимо скачать при первом запуске
# nltk.download("stopwords")
# nltk.download("punkt_tab")
# nltk.download("wordnet")


class TextPreprocessor:
    def __init__(self):
        self.lemmatize = WordNetLemmatizer()
        self.stopwords = set(stopwords.words("english"))

    def __call__(self, text: str):
        return self.preprocess(text)

    def __clean_text(self, text: str):
        """
        Метод обрабатывающий исходный текст.
        В методе из текста удаляются все ссылки, специальные символы и т.д.
        Проводится лемматизация и удаление стоп слов.

        :return: self.text: str - Обработанный текст
        """
        # Удаляем из текста все URL и ссылки
        text = re.sub(r"http\S+", "", text)

        # Приводим все входные данные к нижнему регистру
        text = text.lower()

        # Убираем неалфавитные символы
        text = re.sub("[^a-zA-Z\s]", "", text)

        # Удаляем стоп-слова
        tokens = word_tokenize(text, language="english")  # токенизация
        tokens = [self.lemmatize.lemmatize(token) for token in tokens if not token in self.stopwords]

        return tokens

    @staticmethod
    def __vectorize(tokens):
        """
        Векторизация текста по методу TF-IDF,
        который является статистической мерой для оценки важности слова в документе.

        :return: np.array(dense_matrix) - Возвращаем плотную матрицу в виде numpy-массива
        """

        vectorizer = TfidfVectorizer()

        # Используем vectorizer, чтобы трансформировать текст в векторную форму
        counts_matrix = vectorizer.fit_transform(tokens)

        # Превращаем полученную матрицу в плотную матрицу
        dense_matrix = counts_matrix.todense()

        return np.array(dense_matrix)

    def preprocess(self, text: str):
        tokens = self.__clean_text(text)
        return self.__vectorize(tokens)


if __name__ == "__main__":
    text = Parser("../../JPMartinez2004.pdf").parse()
    preprocessor = TextPreprocessor()
    txt = preprocessor(text)
    print(txt)