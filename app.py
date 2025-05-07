from datetime import datetime, timedelta
from scipy.sparse import hstack
from api_client import APIClient
from data_processor import DataProcessor
from ml_models import LinearRegressionModel, ARIMAModel, SARIMAModel, KNNModel, SVRModel
from text_preprocessor import TextPreprocessor
import joblib

IMPACT_LABELS_RU = {
    "up": "Рост",
    "down": "Падение",
    "neutral": "Нейтрально"
}

class MainApp:
    def __init__(self):
        self.api_client = APIClient()
        self.data_processor = DataProcessor()
        self.multi_model = joblib.load("models/news_model_multi.pkl")
        self.vectorizer = joblib.load("models/news_vectorizer.pkl")
        self.section_encoder = joblib.load("models/news_section_encoder.pkl")
        self.label_encoder = joblib.load("models/news_label_encoder.pkl")
        self.ticker_encoder = joblib.load("models/news_ticker_encoder.pkl")
        self.text_preprocessor = TextPreprocessor()

    def run(self, ticker, from_date, to_date, output_file, threshold=5.0, forecast_days=10):
        print(f"\nЗапуск анализа для {ticker}...")

        stock_data = self.get_and_analyze_stock_data(ticker, from_date, to_date, output_file)
        if stock_data is None:
            return

        significant_days = self.api_client.detect_significant_changes(stock_data, threshold)
        if not significant_days.empty:
            news_data = self.api_client.fetch_news_for_significant_days(ticker, stock_data, threshold)
            self.classify_and_print_news(news_data)

        self.analyze_recent_news(ticker)
        self.run_forecasting_models(ticker, stock_data, forecast_days)

        print(f"\nАнализ {ticker} завершен!")

    def get_and_analyze_stock_data(self, ticker, from_date, to_date, output_file):
        stock_data = self.api_client.fetch_stock_data(ticker, from_date, to_date)
        if stock_data is not None and not stock_data.empty:
            self.data_processor.save_to_csv(stock_data, output_file)

            print(f"\nКраткий анализ цен для {ticker}:")
            self.data_processor.analyze_prices(stock_data, ticker)

            print("\nВизуализация: скользящие средние")
            self.data_processor.calculate_moving_averages(stock_data, ticker)

            print("\nВизуализация: волатильность")
            self.data_processor.calculate_volatility(stock_data, ticker)

            print("\nВизуализация: дневная доходность")
            self.data_processor.calculate_daily_returns(stock_data, ticker)

            return stock_data
        else:
            print("Ошибка: данные по акциям не получены.")
            return None

    def classify_and_print_news(self, news_data):
        if news_data.empty:
            print("Нет новостей, связанных со значительными изменениями.")
            return

        unique_days = news_data["date"].nunique()
        print(f"\nНайдено {len(news_data)} новостей по {unique_days} значимым дням:")

        for _, row in news_data.iterrows():
            title = row["title"]
            url = row["url"]
            processed = self.text_preprocessor.preprocess(title)
            sentiment = TextPreprocessor.analyze_sentiment(title)
            section = row.get("section", "unknown")
            section_code = self.section_encoder.transform([section])[0] if section in self.section_encoder.classes_ else 0

            vector = hstack([
                self.vectorizer.transform([processed]),
                [[sentiment]],
                [[section_code]]
            ])

            pred_encoded = self.multi_model.predict(vector)[0]
            pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
            label = IMPACT_LABELS_RU.get(pred_label, pred_label)

            print(f"{row['date']} | {title} [{label}] [{section}]")
            print(f"   → {url}")

    def run_forecasting_models(self, ticker, stock_data, forecast_days):
        print(f"\nПрогнозирование цен для {ticker} на {forecast_days} дней:")

        for name, model_class in [
            ("Линейная регрессия", LinearRegressionModel),
            ("SVR", SVRModel),
            ("KNN", KNNModel),
            ("ARIMA", ARIMAModel),
            ("SARIMA", SARIMAModel),
        ]:
            print(f"\n{name}:")
            model = model_class(stock_data)
            if name == "ARIMA":
                forecast = model.forecast(order=(1, 1, 1), forecast_days=forecast_days)
            elif name == "SARIMA":
                forecast = model.forecast(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), forecast_days=forecast_days)
            else:
                forecast = model.forecast(forecast_days)
            print(forecast)

    def analyze_recent_news(self, ticker, days=7):
        print(f"\nАнализ новостного фона за последние {days} дней для {ticker}:")

        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=days)

        news_data = self.api_client.fetch_news_from_interfax_range(ticker, start_date, end_date)
        if news_data.empty:
            print("Нет новостей за указанный период.")
            return []

        processed_titles = [self.text_preprocessor.preprocess(t) for t in news_data["title"]]
        sentiments = [TextPreprocessor.analyze_sentiment(t) for t in news_data["title"]]
        sections = news_data.get("section", "unknown").fillna("unknown")

        section_codes = self.section_encoder.transform(sections)

        tfidf_vectors = self.vectorizer.transform(processed_titles)
        vectors = hstack([
            tfidf_vectors,
            [[s] for s in sentiments],
            [[sc] for sc in section_codes]
        ])

        preds_encoded = self.multi_model.predict(vectors)
        preds_labels = self.label_encoder.inverse_transform(preds_encoded)

        news_data["prediction"] = preds_labels
        print(f"\nНайдено {len(news_data)} новостей за последние {days} дней:")

        results = []
        for _, row in news_data.iterrows():
            label = IMPACT_LABELS_RU.get(row["prediction"], row["prediction"])
            print(f"{row['date']} | {row['title']} [{label}]")
            print(f"   → {row['url']}")
            results.append({
                "date": row["date"],
                "title": row["title"],
                "url": row["url"],
                "label": label
            })

        count_up = sum(row["label"] == "Рост" for row in results)
        count_down = sum(row["label"] == "Падение" for row in results)

        print(f"\nИтог: Рост: {count_up} | Падение: {count_down}")
        if count_up > count_down:
            print("Новостной фон положительный — возможен рост.")
        elif count_down > count_up:
            print("Новостной фон отрицательный — возможна просадка.")
        else:
            print("Новостной фон нейтральный.")
        return results
