import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy.sparse import hstack
from imblearn.over_sampling import RandomOverSampler

from text_preprocessor import TextPreprocessor, TFIDFVectorizer


def simplify_class(label: str) -> str:
    if label.startswith("up"):
        return "up"
    elif label.startswith("down"):
        return "down"
    return "neutral"


def main():
    filepath = "data/labeled_news_ALL.csv"
    if not os.path.exists(filepath):
        print(f"Файл не найден: {filepath}")
        return

    df = pd.read_csv(filepath)
    if df.empty or "title" not in df.columns or "impact_class" not in df.columns:
        print("Файл пуст или не содержит нужных колонок (title, impact_class).")
        return

    # Обеспечим наличие поля "ticker"
    if "ticker" not in df.columns:
        df["ticker"] = "unknown"

    df["impact_class_simple"] = df["impact_class"].apply(simplify_class)

    print("Предобработка заголовков...")
    preprocessor = TextPreprocessor()
    processed_titles = [preprocessor.preprocess(t) for t in df["title"]]
    sentiments = [TextPreprocessor.analyze_sentiment(t) for t in df["title"]]

    print("Векторизация текста...")
    vectorizer = TFIDFVectorizer()
    X_text = vectorizer.fit_transform(processed_titles)
    X_sentiment = pd.DataFrame(sentiments, columns=["sentiment"])

    print("Кодирование разделов...")
    section_encoder = LabelEncoder()
    X_section = section_encoder.fit_transform(df["section"].fillna("unknown"))
    X_section = pd.DataFrame(X_section, columns=["section_code"])

    print("Кодирование тикеров...")
    ticker_encoder = LabelEncoder()
    X_ticker = ticker_encoder.fit_transform(df["ticker"].fillna("unknown"))
    X_ticker = pd.DataFrame(X_ticker, columns=["ticker_code"])

    # Собираем полный набор признаков
    X = hstack([X_text, X_sentiment.values, X_section.values, X_ticker.values])
    y = df["impact_class_simple"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Балансировка классов...")
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    unique, counts = np.unique(y_train, return_counts=True)
    print("Распределение после oversampling:")
    for cls, count in zip(label_encoder.inverse_transform(unique), counts):
        print(f"{cls}: {count}")

    print("Обучение модели...")
    model = RandomForestClassifier(
        n_estimators=120, max_depth=10,
        random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Упрощённая модель (3 класса) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 Macro: {f1_score(y_test, y_pred, average='macro'):.3f}")
    print("Классы:", list(label_encoder.classes_))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/news_model_multi.pkl")
    joblib.dump(vectorizer.vectorizer, "models/news_vectorizer.pkl")
    joblib.dump(label_encoder, "models/news_label_encoder.pkl")
    joblib.dump(section_encoder, "models/news_section_encoder.pkl")
    joblib.dump(ticker_encoder, "models/news_ticker_encoder.pkl")

    print("\nМодель и кодировщики сохранены!")


if __name__ == "__main__":
    main()
