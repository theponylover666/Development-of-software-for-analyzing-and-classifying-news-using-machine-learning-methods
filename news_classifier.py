import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy.sparse import hstack
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from text_preprocessor import TFIDFVectorizer

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
        print("Файл пуст или не содержит нужных колонок.")
        return

    if "ticker" not in df.columns:
        df["ticker"] = "unknown"

    df["impact_class_simple"] = df["impact_class"].apply(simplify_class)

    # Удаление коротких или мусорных заголовков
    df["clean_title"] = df["clean_title"].fillna("")
    df = df[df["clean_title"].str.len() > 5]
    df = df[df["num_words"] > 2]

    print(f"После фильтрации: {len(df)} новостей")
    print("\nРаспределение по упрощённым меткам:")
    print(df["impact_class_simple"].value_counts())

    # === Признаки
    print("\nTF-IDF векторизация...")
    vectorizer = TFIDFVectorizer()
    X_text = vectorizer.fit_transform(df["clean_title"])

    print("Добавление числовых признаков...")
    X_sentiment = df[["sentiment"]].fillna(0).values
    X_len = df[["title_len", "num_words"]].fillna(0).values

    print("Кодирование section и ticker...")
    section_encoder = LabelEncoder()
    X_section = section_encoder.fit_transform(df["section"].fillna("unknown")).reshape(-1, 1)

    ticker_encoder = LabelEncoder()
    X_ticker = ticker_encoder.fit_transform(df["ticker"].fillna("unknown")).reshape(-1, 1)

    # Полный вектор признаков
    X = hstack([X_text, X_sentiment, X_len, X_section, X_ticker])
    y = df["impact_class_simple"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Тренировка и тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("\nБалансировка классов...")
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    # === Обучение
    print("\nОбучение XGBoostClassifier...")
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Результаты модели ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 Macro: {f1_score(y_test, y_pred, average='macro'):.3f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # === Очистка старых файлов и сохранение
    print("\nСохраняем модель и кодировщики...")
    os.makedirs("models", exist_ok=True)

    for fname in [
        "news_model_multi.pkl",
        "news_vectorizer.pkl",
        "news_label_encoder.pkl",
        "news_section_encoder.pkl",
        "news_ticker_encoder.pkl"
    ]:
        path = os.path.join("models", fname)
        if os.path.exists(path):
            os.remove(path)

    model.save_model("models/news_model_multi.json")
    joblib.dump(vectorizer.vectorizer, "models/news_vectorizer.pkl")
    joblib.dump(label_encoder, "models/news_label_encoder.pkl")
    joblib.dump(section_encoder, "models/news_section_encoder.pkl")
    joblib.dump(ticker_encoder, "models/news_ticker_encoder.pkl")

    print("✅ Модель и кодировщики успешно сохранены!")

if __name__ == "__main__":
    main()
