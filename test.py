import os
import glob
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.sparse import hstack

from text_preprocessor import TextPreprocessor, TFIDFVectorizer


def update_model(dataset_files, save_dir="models"):
    print("\n=== Обновление общей модели на всех размеченных данных ===")

    all_data = []

    for file in dataset_files:
        df = pd.read_csv(file)
        df = df[df["label"].isin([-1, 1])]
        if not df.empty:
            all_data.append(df)

    if not all_data:
        print("Нет данных для обучения.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"Всего размеченных новостей: {len(full_df)}")

    preprocessor = TextPreprocessor()
    processed_titles = [preprocessor.preprocess(text) for text in full_df["title"]]
    sentiment_scores = [TextPreprocessor.analyze_sentiment(text) for text in processed_titles]

    # Обучаем новый векторизатор (можно сохранять старый, но лучше переобучить)
    vectorizer = TFIDFVectorizer()
    X_text = vectorizer.fit_transform(processed_titles)
    X_sentiment = pd.DataFrame(sentiment_scores)
    X = hstack([X_text, X_sentiment.values])
    y = full_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(classification_report(y_test, y_pred))

    # Сохраняем обновлённую модель
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, "news_model.pkl"))
    joblib.dump(vectorizer.vectorizer, os.path.join(save_dir, "news_vectorizer.pkl"))
    print("✅ Модель и векторизатор обновлены и сохранены.")


def main():
    # Загружаем все кроме labeled_news_ALL.csv
    dataset_files = glob.glob("data/labeled_news_*.csv")
    dataset_files = [f for f in dataset_files]

    update_model(dataset_files)


if __name__ == "__main__":
    main()
