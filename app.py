from api_client import APIClient
from data_processor import DataProcessor
from ml_models import LinearRegressionModel, ARIMAModel, SARIMAModel

class MainApp:
    """Основной класс для запуска приложения."""

    def __init__(self):
        self.api_client = APIClient()
        self.data_processor = DataProcessor()

    def run(self, ticker: str, from_date: str, to_date: str, output_file: str, forecast_days: int = 10):
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
            self.data_processor.analyze_prices(df, ticker)

            # Временной анализ: скользящие средние
            self.data_processor.calculate_moving_averages(df, ticker)

            # Анализ волатильности
            self.data_processor.calculate_volatility(df, ticker)

            # Анализ доходности
            self.data_processor.calculate_daily_returns(df, ticker)

            lr_model = LinearRegressionModel(df)
            lr_forecast = lr_model.forecast(forecast_days)
            print(f"Линейная регрессия для {ticker}: {lr_forecast}")

            ARIMA_model = ARIMAModel(df)
            ARIMA_forecast = ARIMA_model.forecast(order=(1, 1, 1), forecast_days=forecast_days)
            print(f"Авторегрессионное интегрированное с корректированным лагом (ARIMA) для {ticker}: {ARIMA_forecast}")

            SARIMA_model = SARIMAModel(df)
            SARIMA_forecast =SARIMA_model.forecast(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), forecast_days=forecast_days)
            print(f"Авторегрессионное интегрированное сезонное дифференцирование (SARIMA) для {ticker}: {SARIMA_forecast}")

            # Другие алгоритмы (ARIMA, LSTM) могут быть вызваны аналогично
        else:
            print("Не удалось получить данные с API для прогнозирования.")
