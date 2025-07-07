import re
import ssl

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from parser import Parser

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Необходимо скачать при первом запуске
nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")


class TextPreprocessor:
    def __init__(self, text: str):
        self.text = text
        self.lemmatize = WordNetLemmatizer()  # приведение слова к его нормальной форме
        self.stopwords = set(stopwords.words("english"))

    def __delete_stop_words(self):
        """
        Метод, удаляющий из текста стоп-слова.
        Под стоп-словами обычно понимаются артикли, междометия, союзы и т.д.,
        которые не несут смысловой нагрузки.

        :return: text: str - очищенный от стоп-слов слов текст
        """

    def preprocess(self):
        self.text = re.sub("[^a-zA-Z]", " ", self.text)
        self.text = word_tokenize(self.text, language="english")
        print(self.text)


if __name__ == "__main__":
    text = Parser("../../JPMartinez2004.pdf").parse()
    preprocessor = TextPreprocessor(text).preprocess()
