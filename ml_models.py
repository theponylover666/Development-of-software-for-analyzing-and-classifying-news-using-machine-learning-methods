import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

class LinearRegressionModel:
    def __init__(self, data):
        self.data = data

    def forecast(self, forecast_days: int = 10):
        if "CLOSE" not in self.data.columns:
            print("Нет данных для прогноза.")
            return []

        # Подготовка данных
        self.data = self.data.reset_index()
        self.data["DAY"] = np.arange(len(self.data))

        X = self.data[["DAY"]]
        y = self.data["CLOSE"]

        # Обучение модели
        model = LinearRegression()
        model.fit(X, y)

        # Прогноз
        future_days = pd.DataFrame(
            {"DAY": np.arange(len(self.data), len(self.data) + forecast_days)}
        )
        forecast = model.predict(future_days)

        # Визуализация
        self.visualize(X, y, future_days, forecast)

        return forecast

    def visualize(self, X, y, future_days, forecast):
        """
        Визуализация результатов линейной регрессии.
        """
        plt.figure(figsize=(10, 6))

        # График фактических данных
        plt.scatter(X, y, color="blue", label="Фактические цены")
        plt.plot(X, y, color="blue", linestyle="dotted", alpha=0.5)

        # График прогнозируемых данных
        plt.plot(future_days, forecast, color="red", linestyle="--", label="Прогноз")

        plt.title("Прогноз цен с использованием линейной регрессии")
        plt.xlabel("День")
        plt.ylabel("Цена закрытия")
        plt.legend()
        plt.grid(True)
        plt.show()

class ARIMAModel:
    def __init__(self, data: pd.DataFrame):
        """
        Инициализация модели ARIMA.

        :param data: DataFrame с временными рядами. Ожидается колонка 'CLOSE'.
        """
        self.data = data

    def forecast(self, order=(1, 1, 1), forecast_days=10):
        """
        Прогнозирование цен закрытия с использованием ARIMA.

        :param order: Параметры ARIMA (p, d, q).
        :param forecast_days: Количество дней для прогнозирования.
        :return: Список прогнозируемых значений.
        """
        if "CLOSE" not in self.data.columns:
            print("Данные не содержат столбца 'CLOSE'. Прогноз невозможен.")
            return []

        # Подготовка данных
        ts = self.data["CLOSE"].astype(float)

        try:
            # Обучение модели ARIMA
            model = ARIMA(ts, order=order)
            model_fit = model.fit()

            # Прогнозирование
            forecast = model_fit.forecast(steps=forecast_days)
            print("\n### Прогноз ARIMA ###")
            for i, value in enumerate(forecast, 1):
                print(f"День {len(ts) + i}: {value:.2f}")

            # Визуализация прогноза
            self.visualize(ts, forecast, forecast_days)

            return forecast
        except Exception as e:
            print(f"Ошибка при выполнении ARIMA: {e}")
            return []

    def visualize(self, ts, forecast, forecast_days):
        """
        Визуализация фактических данных и прогноза.

        :param ts: Временной ряд фактических данных.
        :param forecast: Прогнозируемые значения.
        :param forecast_days: Количество дней для прогноза.
        """
        plt.figure(figsize=(10, 6))

        # График фактических данных
        plt.scatter(ts.index, ts.values, color="blue", label="Фактические цены")
        plt.plot(ts.index, ts.values, color="blue", linestyle="dotted", alpha=0.5)

        # График прогноза
        future_index = np.arange(len(ts), len(ts) + forecast_days)
        plt.plot(future_index, forecast, label="Прогноз", color="red", linestyle="--")

        plt.title("Прогноз цен с использованием ARIMA")
        plt.xlabel("Дни")
        plt.ylabel("Цена закрытия")
        plt.legend()
        plt.grid(True)
        plt.show()

class SARIMAModel:
    def __init__(self, data: pd.DataFrame):
        """
        Инициализация модели SARIMA.

        :param data: DataFrame с временными рядами. Ожидается колонка 'CLOSE'.
        """
        self.data = data

    def forecast(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), forecast_days=10):
        """
        Прогнозирование цен закрытия с использованием SARIMA.

        :param order: Параметры ARIMA (p, d, q).
        :param seasonal_order: Сезонные параметры SARIMA (P, D, Q, m).
        :param forecast_days: Количество дней для прогнозирования.
        :return: Список прогнозируемых значений.
        """
        if "CLOSE" not in self.data.columns:
            print("Данные не содержат столбца 'CLOSE'. Прогноз невозможен.")
            return []

        # Подготовка данных
        ts = self.data["CLOSE"].astype(float)

        try:
            # Обучение модели SARIMA
            model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit()

            # Прогнозирование
            forecast = model_fit.forecast(steps=forecast_days)
            print("\n### Прогноз SARIMA ###")
            for i, value in enumerate(forecast, 1):
                print(f"День {len(ts) + i}: {value:.2f}")

            # Визуализация прогноза
            self.visualize(ts, forecast, forecast_days)

            return forecast
        except Exception as e:
            print(f"Ошибка при выполнении SARIMA: {e}")
            return []

    def visualize(self, ts, forecast, forecast_days):
        """
        Визуализация фактических данных и прогноза.

        :param ts: Временной ряд фактических данных.
        :param forecast: Прогнозируемые значения.
        :param forecast_days: Количество дней для прогноза.
        """
        plt.figure(figsize=(12, 6))

        # График фактических данных
        plt.scatter(ts.index, ts.values, color="blue", label="Фактические цены")
        plt.plot(ts.index, ts.values, color="blue", linestyle="dotted", alpha=0.5)

        # График прогноза
        future_index = np.arange(len(ts), len(ts) + forecast_days)
        plt.plot(future_index, forecast, label="Прогноз", color="green", linestyle="--")

        plt.title("Прогноз цен с использованием SARIMA")
        plt.xlabel("Дни")
        plt.ylabel("Цена закрытия")
        plt.legend()
        plt.grid(True)
        plt.show()