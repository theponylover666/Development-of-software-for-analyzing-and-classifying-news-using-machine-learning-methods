import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.filterwarnings("ignore", message="Too few observations to estimate starting parameters")

class BaseModel:
    """Базовый класс для прогнозирования, включает общий стиль визуализации."""

    def visualize(self, ts, forecast, forecast_index=None, title="", conf_int_lower=None, conf_int_upper=None):
        plt.figure(figsize=(12, 6))

        ts = ts.copy()
        ts_index = pd.to_datetime(ts.index)

        if forecast_index is None:
            forecast_index = pd.date_range(start=ts_index[-1] + pd.Timedelta(days=1),
                                           periods=len(forecast), freq="B")

        plt.plot(ts_index, ts, label="Исторические данные", color="blue", alpha=0.6)
        plt.plot(forecast_index, forecast, label="Прогноз", color="red", linestyle="--", marker="x")

        if conf_int_lower is not None and conf_int_upper is not None:
            plt.fill_between(forecast_index, conf_int_lower, conf_int_upper,
                             color="pink", alpha=0.3, label="95% интервал")

        plt.title(title, fontsize=14)
        plt.xlabel("Дата", fontsize=12)
        plt.ylabel("Цена закрытия", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        return plt.gcf()


class LinearRegressionModel(BaseModel):
    def __init__(self, stock_data: pd.DataFrame):
        self.stock_data = stock_data.copy()
        if "TRADEDATE" in self.stock_data.columns:
            self.stock_data["TRADEDATE"] = pd.to_datetime(self.stock_data["TRADEDATE"])
            self.stock_data.set_index("TRADEDATE", inplace=True)
            self.stock_data = self.stock_data[~self.stock_data.index.duplicated()]
            self.stock_data = self.stock_data.asfreq("B")
            self.stock_data["CLOSE"] = self.stock_data["CLOSE"].ffill()
        else:
            self.stock_data.index = pd.date_range(start="2021-01-01", periods=len(self.stock_data), freq="B")

    def preprocess_data(self):
        self.stock_data["DAY"] = np.arange(len(self.stock_data))
        return self.stock_data

    def forecast(self, forecast_days: int = 10, return_data=False):
        stock_data = self.preprocess_data()
        X = stock_data[["DAY"]]
        y = stock_data["CLOSE"]

        if X.empty or y.empty or y.isnull().any():
            print("Ошибка: переданы пустые массивы X или y или есть пропуски.")
            return None if not return_data else []

        model = LinearRegression()
        model.fit(X, y)

        future_index = np.linspace(X["DAY"].max() + 1, X["DAY"].max() + forecast_days, forecast_days)
        future_days = pd.DataFrame({"DAY": future_index})
        forecast = model.predict(future_days)

        forecast_dates = pd.date_range(start=self.stock_data.index[-1] + pd.Timedelta(days=1),
                                       periods=forecast_days, freq="B")
        forecast_data = [{"date": str(date.date()), "value": float(val)} for date, val in zip(forecast_dates, forecast)]

        if return_data:
            return forecast_data
        else:
            return self.visualize(y, forecast, forecast_index=forecast_dates,
                                  title="Прогноз цен: Линейная регрессия")


class ARIMAModel(BaseModel):
    def __init__(self, stock_data: pd.DataFrame):
        self.stock_data = stock_data.copy()
        if "TRADEDATE" in self.stock_data.columns:
            self.stock_data["TRADEDATE"] = pd.to_datetime(self.stock_data["TRADEDATE"])
            self.stock_data.set_index("TRADEDATE", inplace=True)
            self.stock_data = self.stock_data[~self.stock_data.index.duplicated()]
            self.stock_data = self.stock_data.asfreq("B")
        else:
            self.stock_data.index = pd.date_range(start="2021-01-01", periods=len(self.stock_data), freq="B")

    def forecast(self, order=(1, 1, 1), forecast_days=10, return_data=False):
        ts = self.stock_data["CLOSE"].astype(float)

        try:
            model = ARIMA(ts, order=order)
            model_fit = model.fit()

            pred = model_fit.get_forecast(steps=forecast_days)
            forecast = pred.predicted_mean
            conf_int = pred.conf_int(alpha=0.05)

            forecast_dates = pd.date_range(start=self.stock_data.index[-1] + pd.Timedelta(days=1),
                                           periods=forecast_days, freq="B")

            if return_data:
                return [{"date": str(date.date()), "value": float(val)} for date, val in zip(forecast_dates, forecast)]
            else:
                self.visualize(ts, forecast, forecast_index=forecast_dates,
                               conf_int_lower=conf_int.iloc[:, 0],
                               conf_int_upper=conf_int.iloc[:, 1],
                               title="Прогноз ARIMA")
                return list(forecast)

        except Exception as e:
            print(f"Ошибка в ARIMA: {e}")
            return []


class SARIMAModel(BaseModel):
    def __init__(self, stock_data: pd.DataFrame):
        self.stock_data = stock_data.copy()
        if "TRADEDATE" in self.stock_data.columns:
            self.stock_data["TRADEDATE"] = pd.to_datetime(self.stock_data["TRADEDATE"])
            self.stock_data.set_index("TRADEDATE", inplace=True)
            self.stock_data = self.stock_data[~self.stock_data.index.duplicated()]
            self.stock_data = self.stock_data.asfreq("B")
        else:
            self.stock_data.index = pd.date_range(start="2021-01-01", periods=len(self.stock_data), freq="B")

    def forecast(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), forecast_days=10, return_data=False):
        ts = self.stock_data["CLOSE"].astype(float)

        try:
            if len(ts) < 30:
                seasonal_order = (0, 0, 0, 0)

            model = SARIMAX(ts,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            model_fit = model.fit(method='lbfgs', disp=0)

            pred = model_fit.get_forecast(steps=forecast_days)
            forecast = pred.predicted_mean
            conf_int = pred.conf_int(alpha=0.05)

            forecast_dates = pd.date_range(start=self.stock_data.index[-1] + pd.Timedelta(days=1),
                                           periods=forecast_days, freq="B")

            if return_data:
                return [{"date": str(date.date()), "value": float(val)} for date, val in zip(forecast_dates, forecast)]
            else:
                self.visualize(ts, forecast, forecast_index=forecast_dates,
                               conf_int_lower=conf_int.iloc[:, 0],
                               conf_int_upper=conf_int.iloc[:, 1],
                               title="Прогноз SARIMA")
                return list(forecast)

        except Exception as e:
            print(f"Ошибка в SARIMA: {e}")
            return []


class SVRModel(BaseModel):
    def __init__(self, stock_data: pd.DataFrame):
        self.stock_data = stock_data.copy()
        self.stock_data["TRADEDATE"] = pd.to_datetime(self.stock_data["TRADEDATE"])
        self.stock_data.drop_duplicates(subset=["TRADEDATE"], inplace=True)
        self.stock_data.set_index("TRADEDATE", inplace=True)
        self.stock_data = self.stock_data.asfreq("B")
        self.stock_data["CLOSE"] = self.stock_data["CLOSE"].ffill()
        self.stock_data["DAY"] = np.arange(len(self.stock_data))

    def forecast(self, forecast_days: int = 10, return_data=False):
        X = self.stock_data[["DAY"]]
        y = self.stock_data["CLOSE"]

        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        model.fit(X, y)

        future_days = np.arange(len(self.stock_data), len(self.stock_data) + forecast_days).reshape(-1, 1)
        forecast = model.predict(future_days)

        forecast_index = pd.date_range(start=self.stock_data.index[-1] + pd.Timedelta(days=1),
                                       periods=forecast_days, freq="B")

        if return_data:
            return [{"date": str(date.date()), "value": float(val)} for date, val in zip(forecast_index, forecast)]
        else:
            return self.visualize(y, forecast, forecast_index=forecast_index, title="SVR: прогноз цен")


class KNNModel(BaseModel):
    def __init__(self, stock_data: pd.DataFrame):
        self.stock_data = stock_data.copy()
        self.stock_data["TRADEDATE"] = pd.to_datetime(self.stock_data["TRADEDATE"])
        self.stock_data.drop_duplicates(subset=["TRADEDATE"], inplace=True)
        self.stock_data.set_index("TRADEDATE", inplace=True)
        self.stock_data = self.stock_data.asfreq("B")
        self.stock_data["CLOSE"] = self.stock_data["CLOSE"].ffill()
        self.stock_data["DAY"] = np.arange(len(self.stock_data))

    def forecast(self, forecast_days: int = 10, return_data=False):
        X = self.stock_data[["DAY"]]
        y = self.stock_data["CLOSE"]

        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X, y)

        future_days = np.arange(len(self.stock_data), len(self.stock_data) + forecast_days).reshape(-1, 1)
        forecast = model.predict(future_days)

        forecast_index = pd.date_range(start=self.stock_data.index[-1] + pd.Timedelta(days=1),
                                       periods=forecast_days, freq="B")

        if return_data:
            return [{"date": str(date.date()), "value": float(val)} for date, val in zip(forecast_index, forecast)]
        else:
            return self.visualize(y, forecast, forecast_index=forecast_index, title="KNN: прогноз цен")
