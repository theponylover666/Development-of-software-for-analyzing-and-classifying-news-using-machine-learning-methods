from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
from sklearn.feature_extraction.text import TfidfVectorizer


POSITIVE_WORDS = {
    "прибыль", "рост", "увеличение", "развитие", "рекорд", "дивиденды", "партнёрство",
    "успех", "инвестиции", "поддержка", "инновации", "прорыв", "премия", "положительно",
    "продвижение", "приобретение", "повышение", "экспорт", "улучшение", "превышение"
}

NEGATIVE_WORDS = {
    "убыток", "санкции", "снижение", "отставка", "сокращение", "штраф", "кризис",
    "падение", "убытки", "провал", "рецессия", "авария", "долг", "убыль", "дефолт",
    "арест", "банкротство", "инфляция", "убыточный", "отрицательно", "риски"
}


class TextPreprocessor:
    def __init__(self):
        self.segmenter = Segmenter()
        self.embedding = NewsEmbedding()
        self.tagger = NewsMorphTagger(self.embedding)
        self.morph_vocab = MorphVocab()

    def preprocess(self, text: str) -> str:
        """
        Токенизация, морфоанализ и лемматизация текста.
        Возвращает строку из нормализованных слов (лемм).
        """
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.tagger)

        lemmas = []
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            if token.pos in {"NOUN", "VERB", "ADJ", "ADV"}:
                lemmas.append(token.lemma.lower())

        return " ".join(lemmas)

    @staticmethod
    def analyze_sentiment(text: str) -> int:
        """
        Анализ наивной тональности на основе словаря.
        Возвращает: 1 (позитив), -1 (негатив), 0 (нейтрально)
        """
        words = text.lower().split()
        score = 0
        for word in words:
            if word in POSITIVE_WORDS:
                score += 1
            elif word in NEGATIVE_WORDS:
                score -= 1

        return 1 if score > 0 else -1 if score < 0 else 0

    def __repr__(self):
        return "<TextPreprocessor (Natasha + TF-IDF + sentiment)>"


class TFIDFVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, texts):
        """Обучает TF-IDF и возвращает матрицу признаков."""
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """Трансформирует новые тексты в TF-IDF-векторы."""
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        """Список признаков."""
        return self.vectorizer.get_feature_names_out()

    def __repr__(self):
        return f"<TFIDFVectorizer ({len(self.vectorizer.vocabulary_)} terms)>"
