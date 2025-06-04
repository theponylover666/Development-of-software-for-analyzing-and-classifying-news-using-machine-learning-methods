import pandas as pd
import matplotlib.pyplot as plt

class DataProcessor:
    """
    Класс для анализа и визуализации биржевых данных.
    """

    @staticmethod
    def save_to_csv(data: pd.DataFrame, filename: str):
        """
        Сохраняет DataFrame в CSV-файл, если он не пуст.
        """
        if data.empty:
            print("Нет данных для сохранения.")
            return
        data.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"Данные сохранены в файл: {filename}")

    @staticmethod
    def analyze_prices(data: pd.DataFrame, ticker: str):
        """
        Выводит основные статистики по столбцу 'CLOSE'.
        Также рассчитывает процентное изменение, если доступна колонка 'PREVWAPRICE'.
        """
        if "CLOSE" not in data.columns:
            print(f"Ошибка: отсутствует колонка 'CLOSE' для {ticker}")
            return

        stats = data["CLOSE"].describe()
        print(f"\nАнализ ({ticker}):")
        print(f"- Дней: {int(stats['count'])}")
        print(f"- Средняя цена: {stats['mean']:.2f}")
        print(f"- Мин: {stats['min']:.2f} | Макс: {stats['max']:.2f}")
        print(f"- Медиана: {stats['50%']:.2f}")
        print(f"- Стандартное отклонение: {stats['std']:.2f}")

        if "PREVWAPRICE" in data.columns:
            data["PRICE_CHANGE_%"] = (
                (data["CLOSE"] - data["PREVWAPRICE"]) / data["PREVWAPRICE"] * 100
            )
        else:
            print("Предупреждение: 'PREVWAPRICE' отсутствует — изменение цены не рассчитано.")

    @staticmethod
    def calculate_moving_averages(data: pd.DataFrame, ticker: str):
        """
        Строит график скользящих средних: SMA 10 и SMA 50.
        """
        if "CLOSE" not in data.columns:
            return None

        data["SMA_10"] = data["CLOSE"].rolling(window=10).mean()
        data["SMA_50"] = data["CLOSE"].rolling(window=50).mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["TRADEDATE"], data["CLOSE"], label="Цена закрытия", color="blue")
        ax.plot(data["TRADEDATE"], data["SMA_10"], label="SMA 10", color="green")
        ax.plot(data["TRADEDATE"], data["SMA_50"], label="SMA 50", color="red")

        ax.set_title(f"{ticker} — Скользящие средние")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    @staticmethod
    def calculate_volatility(data: pd.DataFrame, ticker: str):
        """
        Строит график 10-дневной волатильности (стандартного отклонения).
        """
        if "CLOSE" not in data.columns:
            return None

        data["VOLATILITY"] = data["CLOSE"].rolling(window=10).std()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["TRADEDATE"], data["VOLATILITY"], label="Волатильность (10 дней)", color="purple")

        ax.set_title(f"{ticker} — Волатильность")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    @staticmethod
    def calculate_daily_returns(data: pd.DataFrame, ticker: str):
        """
        Строит гистограмму дневной доходности в процентах.
        """
        if "CLOSE" not in data.columns:
            return None

        data["DAILY_RETURN"] = data["CLOSE"].pct_change() * 100

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data["DAILY_RETURN"].dropna(), bins=20, color="orange", edgecolor="black")

        ax.set_title(f"{ticker} — Распределение дневной доходности")
        ax.grid(True)
        plt.tight_layout()
        return fig
