import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Необходимо скачать при первом запуске
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)


class TextPreprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words("english"))

    def __call__(self, text: str):
        return self.__clean_text(text)

    def __clean_text(self, text: str):
        """
        Метод обрабатывающий исходный текст.
        В методе из текста удаляются все ссылки, специальные символы и т.д.
        Проводится лемматизация и удаление стоп слов.

        :return: tokens: list - Возвращает список токенов
        """
        # Удаляем из текста все URL и ссылки
        text = re.sub(r"http\S+", "", text)

        # Приводим все входные данные к нижнему регистру
        text = text.lower()

        # Убираем неалфавитные символы
        text = re.sub("[^a-zA-Z\s]", "", text)

        # Токенизируем с помощью библиотеки NLTK
        tokens = word_tokenize(text, language="english")  # токенизация

        # Удаляем стоп-слова
        tokens = [token for token in tokens if token not in self.stopwords]
        return tokens
