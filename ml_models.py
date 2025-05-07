import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from utils import add_features
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.filterwarnings("ignore", message="Too few observations to estimate starting parameters")

class BaseModel:
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

    def tune_model(self, model, param_grid, randomized=False, n_iter=10):
        tscv = TimeSeriesSplit(n_splits=5)
        if randomized:
            search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=tscv,
                                        scoring="neg_mean_squared_error", random_state=42)
        else:
            search = GridSearchCV(model, param_grid, cv=tscv, scoring="neg_mean_squared_error")
        search.fit(self.X, self.y)
        return search.best_estimator_

class LinearRegressionModel(BaseModel):
    def __init__(self, stock_data: pd.DataFrame, day_offset=0):
        self.stock_data = stock_data.copy()
        self.day_offset = day_offset

        self.stock_data["TRADEDATE"] = pd.to_datetime(self.stock_data["TRADEDATE"])
        self.stock_data.drop_duplicates(subset="TRADEDATE", keep="last", inplace=True)
        self.stock_data.set_index("TRADEDATE", inplace=True)
        self.stock_data = self.stock_data.asfreq("B")
        self.stock_data["CLOSE"] = self.stock_data["CLOSE"].ffill()

        self.stock_data = add_features(self.stock_data)
        self.X = self.stock_data[["MA3", "MA5", "EMA10", "STD_5", "RETURN"]]
        self.y = self.stock_data["CLOSE"]

    def forecast(self, forecast_days: int = 10, return_data=False):
        model = LinearRegression()
        model.fit(self.X, self.y)

        preds = []
        df = self.stock_data.copy()

        for _ in range(forecast_days):
            last_row = df.iloc[-1:].copy()
            features = last_row[["MA3", "MA5", "EMA10", "STD_5", "RETURN"]]
            pred = model.predict(features)[0]
            preds.append(pred)

            new_row = last_row.copy()
            new_row["CLOSE"] = pred
            df = pd.concat([df, new_row])
            df = add_features(df)

        forecast_index = pd.date_range(start=self.stock_data.index[-1] + pd.Timedelta(days=1),
                                       periods=forecast_days, freq="B")

        if return_data:
            return [{"date": str(date.date()), "value": float(val)} for date, val in zip(forecast_index, preds)]
        else:
            return self.visualize(self.y, preds, forecast_index, title="Linear Regression с признаками")

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

        if len(ts) < 30:
            print("ARIMA: недостаточно данных (нужно ≥30 точек)")
            return None if not return_data else []

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
                return self.visualize(ts, forecast, forecast_index=forecast_dates,
                                      conf_int_lower=conf_int.iloc[:, 0],
                                      conf_int_upper=conf_int.iloc[:, 1],
                                      title="Прогноз ARIMA")
        except Exception as e:
            print(f"ARIMA: ошибка — {e}")
            return None if not return_data else []

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

        if len(ts) < 30:
            print("SARIMA: недостаточно данных (нужно ≥30 точек)")
            return None if not return_data else []

        try:
            model = SARIMAX(ts,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            model_fit = model.fit(disp=0)

            pred = model_fit.get_forecast(steps=forecast_days)
            forecast = pred.predicted_mean
            conf_int = pred.conf_int(alpha=0.05)

            forecast_dates = pd.date_range(start=self.stock_data.index[-1] + pd.Timedelta(days=1),
                                           periods=forecast_days, freq="B")

            if return_data:
                return [{"date": str(date.date()), "value": float(val)} for date, val in zip(forecast_dates, forecast)]
            else:
                return self.visualize(ts, forecast, forecast_index=forecast_dates,
                                      conf_int_lower=conf_int.iloc[:, 0],
                                      conf_int_upper=conf_int.iloc[:, 1],
                                      title="Прогноз SARIMA")
        except Exception as e:
            print(f"SARIMA: ошибка — {e}")
            return None if not return_data else []

class SVRModel(BaseModel):
    def __init__(self, stock_data: pd.DataFrame, day_offset=0):
        self.stock_data = stock_data.copy()
        self.day_offset = day_offset

        self.stock_data["TRADEDATE"] = pd.to_datetime(self.stock_data["TRADEDATE"])
        self.stock_data.drop_duplicates(subset="TRADEDATE", keep="last", inplace=True)
        self.stock_data.set_index("TRADEDATE", inplace=True)
        self.stock_data = self.stock_data.asfreq("B")
        self.stock_data["CLOSE"] = self.stock_data["CLOSE"].ffill()

        self.stock_data = add_features(self.stock_data)
        self.X = self.stock_data[["MA3", "MA5", "EMA10", "STD_5", "RETURN"]]
        self.y = self.stock_data["CLOSE"]
    def forecast(self, forecast_days: int = 10, return_data=False, tune=False):
        if tune:
            param_grid = {
                "C": [1, 10, 100],
                "gamma": [0.01, 0.1, 1],
                "epsilon": [0.01, 0.1, 0.5]
            }
            model = self.tune_model(SVR(kernel='rbf'), param_grid)
        else:
            model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
            model.fit(self.X, self.y)

        preds = []
        df = self.stock_data.copy()

        for _ in range(forecast_days):
            last_row = df.iloc[-1:].copy()
            features = last_row[["MA3", "MA5", "EMA10", "STD_5", "RETURN"]]
            pred = model.predict(features)[0]
            preds.append(pred)

            new_row = last_row.copy()
            new_row["CLOSE"] = pred
            df = pd.concat([df, new_row])
            df = add_features(df)

        forecast_index = pd.date_range(start=self.stock_data.index[-1] + pd.Timedelta(days=1),
                                       periods=forecast_days, freq="B")

        if return_data:
            return [{"date": str(date.date()), "value": float(val)} for date, val in zip(forecast_index, preds)]
        else:
            return self.visualize(self.y, preds, forecast_index,
                                  title="Прогноз SVR (с гиперпараметрами)" if tune else "Прогноз SVR")

class KNNModel(SVRModel):
    def forecast(self, forecast_days: int = 10, return_data=False):
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(self.X, self.y)

        preds = []
        df = self.stock_data.copy()

        for _ in range(forecast_days):
            last_row = df.iloc[-1:].copy()
            features = last_row[["MA3", "MA5", "EMA10", "STD_5", "RETURN"]]
            pred = model.predict(features)[0]
            preds.append(pred)

            new_row = last_row.copy()
            new_row["CLOSE"] = pred
            df = pd.concat([df, new_row])
            df = add_features(df)

        forecast_index = pd.date_range(start=self.stock_data.index[-1] + pd.Timedelta(days=1),
                                       periods=forecast_days, freq="B")

        if return_data:
            return [{"date": str(date.date()), "value": float(val)} for date, val in zip(forecast_index, preds)]
        else:
            return self.visualize(self.y, preds, forecast_index, title="Прогноз KNN")

class XGBoostModel(BaseModel):
    def __init__(self, stock_data: pd.DataFrame, day_offset=0):
        self.stock_data = stock_data.copy()
        self.day_offset = day_offset

        self.stock_data["TRADEDATE"] = pd.to_datetime(self.stock_data["TRADEDATE"])
        self.stock_data.drop_duplicates(subset="TRADEDATE", keep="last", inplace=True)
        self.stock_data.set_index("TRADEDATE", inplace=True)
        self.stock_data = self.stock_data.asfreq("B")
        self.stock_data["CLOSE"] = self.stock_data["CLOSE"].ffill()

        self.stock_data = add_features(self.stock_data)
        self.X = self.stock_data[["MA3", "MA5", "EMA10", "STD_5", "RETURN"]]
        self.y = self.stock_data["CLOSE"]
    def forecast(self, forecast_days: int = 10, return_data=False, tune=False):
        if tune:
            param_grid = {
                "n_estimators": [50, 100, 150],
                "max_depth": [3, 4, 5],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 1],
                "colsample_bytree": [0.7, 1]
            }
            model = self.tune_model(XGBRegressor(random_state=42), param_grid, randomized=True, n_iter=10)
        else:
            model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
            model.fit(self.X, self.y)

        preds = []
        df = self.stock_data.copy()

        for _ in range(forecast_days):
            last_row = df.iloc[-1:].copy()
            features = last_row[["MA3", "MA5", "EMA10", "STD_5", "RETURN"]]
            pred = model.predict(features)[0]
            preds.append(pred)

            new_row = last_row.copy()
            new_row["CLOSE"] = pred
            df = pd.concat([df, new_row])
            df = add_features(df)

        forecast_index = pd.date_range(start=self.stock_data.index[-1] + pd.Timedelta(days=1),
                                       periods=forecast_days, freq="B")

        if return_data:
            return [{"date": str(date.date()), "value": float(val)} for date, val in zip(forecast_index, preds)]
        else:
            return self.visualize(self.y, preds, forecast_index,
                                  title="Прогноз XGBoost (с гиперпараметрами)" if tune else "Прогноз XGBoost")

