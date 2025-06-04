from datetime import datetime, timedelta
from scipy.sparse import hstack
from api_client import APIClient
from data_processor import DataProcessor
from ml_models import LinearRegressionModel, ARIMAModel, SARIMAModel, KNNModel, SVRModel
from text_preprocessor import TextPreprocessor
import joblib
from xgboost import XGBClassifier

# Отображение меток на русском языке
IMPACT_LABELS_RU = {
    "up": "Рост",
    "down": "Падение",
    "neutral": "Нейтрально"
}

class MainApp:
    def __init__(self):
        # Инициализация компонентов и загрузка моделей
        self.api_client = APIClient()
        self.data_processor = DataProcessor()
        self.text_preprocessor = TextPreprocessor()
        self.multi_model = XGBClassifier()
        self.multi_model.load_model("models/news_model_multi.json")

        self.vectorizer = joblib.load("models/news_vectorizer.pkl")
        self.section_encoder = joblib.load("models/news_section_encoder.pkl")
        self.label_encoder = joblib.load("models/news_label_encoder.pkl")
        self.ticker_encoder = joblib.load("models/news_ticker_encoder.pkl")

    def run(self, ticker, from_date, to_date, output_file, threshold=5.0, forecast_days=10):
        """
        Основной метод запуска анализа: акции, новости, прогнозирование.
        """
        stock_data = self.get_and_analyze_stock_data(ticker, from_date, to_date, output_file)
        if stock_data is None:
            return

        significant_days = self.api_client.detect_significant_changes(stock_data, threshold)
        if not significant_days.empty:
            news_data = self.api_client.fetch_news_for_significant_days(ticker, stock_data, threshold)
            self.classify_and_print_news(news_data)

        self.analyze_recent_news(ticker)
        self.run_forecasting_models(ticker, stock_data, forecast_days)

    def get_and_analyze_stock_data(self, ticker, from_date, to_date, output_file):
        """
        Получение и базовый анализ данных по акциям.
        """
        stock_data = self.api_client.fetch_stock_data(ticker, from_date, to_date)
        if stock_data is None or stock_data.empty:
            print("Ошибка: данные по акциям не получены.")
            return None

        self.data_processor.save_to_csv(stock_data, output_file)
        self.data_processor.analyze_prices(stock_data, ticker)
        self.data_processor.calculate_moving_averages(stock_data, ticker)
        self.data_processor.calculate_volatility(stock_data, ticker)
        self.data_processor.calculate_daily_returns(stock_data, ticker)
        return stock_data

    def classify_and_print_news(self, news_data):
        """
        Классификация новостей и вывод в консоль.
        """
        if news_data.empty:
            return

        for _, row in news_data.iterrows():
            title = row["title"]
            section = row.get("section", "unknown")
            ticker = row.get("ticker", "unknown")

            processed = self.text_preprocessor.preprocess(title)
            sentiment = self.text_preprocessor.analyze_sentiment(title)
            section_code = self.section_encoder.transform([section])[0] if section in self.section_encoder.classes_ else 0
            ticker_code = self.ticker_encoder.transform([ticker])[0] if ticker in self.ticker_encoder.classes_ else 0

            vector = hstack([
                self.vectorizer.transform([processed]),
                [[sentiment]],
                [[len(title), len(processed.split())]],
                [[section_code]],
                [[ticker_code]]
            ])

            pred_encoded = self.multi_model.predict(vector)[0]
            pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
            label = IMPACT_LABELS_RU.get(pred_label, pred_label)

            print(f"{row['date']} | {title} [{label}] [{section}]")
            print(f"   → {row['url']}")

    def run_forecasting_models(self, ticker, stock_data, forecast_days):
        """
        Запуск моделей прогнозирования цен акций.
        """
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
                model.forecast(order=(1, 1, 1), forecast_days=forecast_days)
            elif name == "SARIMA":
                model.forecast(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), forecast_days=forecast_days)
            else:
                model.forecast(forecast_days)

    def analyze_recent_news(self, ticker, days=7):
        """
        Анализ и классификация новостей за последние N дней.
        """
        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=days)

        news_data = self.api_client.fetch_news_from_interfax_range(ticker, start_date, end_date)
        if news_data.empty:
            return []

        processed_titles = [self.text_preprocessor.preprocess(t) for t in news_data["title"]]
        sentiments = [TextPreprocessor.analyze_sentiment(t) for t in news_data["title"]]
        sections = news_data.get("section", "unknown").fillna("unknown")
        section_codes = self.section_encoder.transform(sections)

        tfidf_vectors = self.vectorizer.transform(processed_titles)
        title_lens = [len(t) for t in news_data["title"]]
        num_words = [len(p.split()) for p in processed_titles]
        tickers = news_data.get("ticker", "unknown").fillna("unknown")
        ticker_codes = self.ticker_encoder.transform(tickers)

        vectors = hstack([
            tfidf_vectors,
            [[s] for s in sentiments],
            [[tl, nw] for tl, nw in zip(title_lens, num_words)],
            [[sc] for sc in section_codes],
            [[tc] for tc in ticker_codes]
        ])

        preds_encoded = self.multi_model.predict(vectors)
        preds_labels = self.label_encoder.inverse_transform(preds_encoded)
        news_data["prediction"] = preds_labels

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

        return results

    def test_manual_prediction(self, title, section="Экономика", ticker="SBER"):
        """
        Ручной тест классификации новости по заголовку.
        """
        processed = self.text_preprocessor.preprocess(title)
        sentiment = self.text_preprocessor.analyze_sentiment(title)
        section_code = self.section_encoder.transform([section])[0] if section in self.section_encoder.classes_ else 0
        ticker_code = self.ticker_encoder.transform([ticker])[0] if ticker in self.ticker_encoder.classes_ else 0

        vector = hstack([
            self.vectorizer.transform([processed]),
            [[sentiment]],
            [[len(title), len(processed.split())]],
            [[section_code]],
            [[ticker_code]]
        ])

        pred_encoded = self.multi_model.predict(vector)[0]
        label = self.label_encoder.inverse_transform([pred_encoded])[0]
        print(f"'{title}' → {IMPACT_LABELS_RU.get(label, label)}")
