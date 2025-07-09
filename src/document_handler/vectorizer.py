import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x, # текст уже токенизирован
            preprocessor=lambda x: x, # текст уже обработан
            token_pattern=None,
            lowercase=False # текст уже приведен к одному регистру
        )

    def fit(self, texts):
        """texts – список списков токенов (уже очищенных)."""
        self.vectorizer.fit(texts)

    def transform(self, tokens):
        """Возвращает 1-D вектор фиксированной длины."""
        vec = self.vectorizer.transform([tokens]).todense()
        return np.asarray(vec, dtype=np.float32).squeeze(0)
