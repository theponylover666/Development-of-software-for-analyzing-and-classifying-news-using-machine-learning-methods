import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy.sparse import hstack

from text_preprocessor import TextPreprocessor, TFIDFVectorizer


def simplify_class(label: str) -> str:
    """Преобразует мультиклассовую метку в упрощённую (3 класса)."""
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

    # === Упрощённые метки ===
    df["impact_class_simple"] = df["impact_class"].apply(simplify_class)

    # === Предобработка текста ===
    print("Предобработка заголовков...")
    preprocessor = TextPreprocessor()
    processed_titles = [preprocessor.preprocess(t) for t in df["title"]]
    sentiments = [TextPreprocessor.analyze_sentiment(t) for t in df["title"]]

    # === TF-IDF ===
    print("Векторизация текста...")
    vectorizer = TFIDFVectorizer()
    X_text = vectorizer.fit_transform(processed_titles)
    X_sentiment = pd.DataFrame(sentiments, columns=["sentiment"])

    # === Кодировка раздела новости ===
    print("Кодирование разделов...")
    section_encoder = LabelEncoder()
    X_section = section_encoder.fit_transform(df["section"].fillna("unknown"))
    X_section = pd.DataFrame(X_section, columns=["section_code"])

    # === Объединение всех признаков ===
    X = hstack([X_text, X_sentiment.values, X_section.values])
    y = df["impact_class_simple"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # === Разделение на обучение и тест ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # === Обучение модели ===
    print("Обучение модели...")
    model = RandomForestClassifier(
        n_estimators=120, max_depth=10,
        random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # === Отчёт ===
    print("\n=== Упрощённая модель (3 класса) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 Macro: {f1_score(y_test, y_pred, average='macro'):.3f}")
    print("Классы:", list(label_encoder.classes_))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # === Сохранение модели и кодировщиков ===
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/news_model_multi.pkl")
    joblib.dump(vectorizer.vectorizer, "models/news_vectorizer.pkl")
    joblib.dump(label_encoder, "models/news_label_encoder.pkl")
    joblib.dump(section_encoder, "models/news_section_encoder.pkl")

    print("\nУпрощённая модель и кодировщики сохранены в папку: models/")

if __name__ == "__main__":
    main()
