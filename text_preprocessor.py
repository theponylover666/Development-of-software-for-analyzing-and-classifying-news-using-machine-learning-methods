from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
from sklearn.feature_extraction.text import TfidfVectorizer

# Расширенный словарь положительных слов (умеренно)
POSITIVE_WORDS = {
    "прибыль", "рост", "увеличение", "развитие", "рекорд", "дивиденды",
    "партнёрство", "успех", "инвестиции", "поддержка", "инновации", "прорыв",
    "премия", "приобретение", "повышение", "экспорт", "улучшение", "превышение",
    "укрепление", "стабильность", "одобрение", "прогресс"
}

# Расширенный словарь негативных слов (приоритетный)
NEGATIVE_WORDS = {
    "убыток", "санкции", "снижение", "отставка", "сокращение", "штраф", "кризис",
    "падение", "убытки", "провал", "рецессия", "авария", "долг", "убыль", "дефолт",
    "арест", "банкротство", "инфляция", "убыточный", "отрицательно", "риски",
    "задержка", "обвал", "шторм", "пожар", "потери", "злоупотребление", "разбирательство",
    "захват", "протест", "убийство", "расправа", "катастрофа", "убой", "скандал",
    "суд", "расследование", "паника", "проблема", "угроза", "недовольство", "дефицит"
}


class TextPreprocessor:
    """
    Обёртка для лемматизации и анализа тональности текста.
    Использует Natasha и простой словарь тональности.
    """

    def __init__(self):
        self.segmenter = Segmenter()
        self.embedding = NewsEmbedding()
        self.tagger = NewsMorphTagger(self.embedding)
        self.morph_vocab = MorphVocab()

    def preprocess(self, text: str) -> str:
        """
        Токенизация и лемматизация текста.
        Возвращает строку нормализованных слов (лемм).
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
        Анализ тональности текста с усилением негатива.
        Возвращает:
            1 — позитив (если набрано ≥2 баллов),
           -1 — негатив (любая сумма ≤ -1),
            0 — нейтрально
        """
        words = text.lower().split()
        score = 0
        for word in words:
            if word in POSITIVE_WORDS:
                score += 1
            elif word in NEGATIVE_WORDS:
                score -= 2

        if score >= 2:
            return 1
        elif score <= -1:
            return -1
        else:
            return 0

    def __repr__(self):
        return "<TextPreprocessor (Natasha + TF-IDF + sentiment)>"


class TFIDFVectorizer:
    """
    Обёртка для TF-IDF векторизатора на основе sklearn.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, texts):
        """
        Обучает TF-IDF и преобразует входной список текстов.
        """
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """
        Преобразует новые тексты в TF-IDF векторы.
        """
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        """
        Возвращает список признаков (термов).
        """
        return self.vectorizer.get_feature_names_out()

    def __repr__(self):
        return f"<TFIDFVectorizer ({len(self.vectorizer.vocabulary_)} terms)>"
