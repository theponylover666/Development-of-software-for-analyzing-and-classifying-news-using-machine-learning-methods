from api_client import APIClient
from data_processor import DataProcessor

class MainApp:
    """Основной класс для запуска приложения."""

    def __init__(self):
        self.api_client = APIClient()
        self.data_processor = DataProcessor()

    def run(self, ticker: str, from_date: str, to_date: str, output_file: str):
        """
        Основной процесс: получение данных, анализ и визуализация.
        """
        print(f"Сбор данных для тикера {ticker} с {from_date} по {to_date}...")

        # Получение данных с API
        df = self.api_client.fetch_stock_data(ticker, from_date, to_date)

        if df is not None and not df.empty:
            # Сохранение в CSV
            self.data_processor.save_to_csv(df, output_file)

            # Анализ данных
            self.data_processor.analyze_prices(df)
            self.data_processor.analyze_liquidity(df)
            self.data_processor.classify_parameters(df)
            self.data_processor.compare_companies(df)

            # Визуализация данных
            self.data_processor.visualize_data(df, ticker)
        else:
            print("Не удалось получить данные с API.")
