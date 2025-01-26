import pandas as pd
import matplotlib.pyplot as plt

class DataProcessor:
    """Класс для обработки и анализа данных."""

    @staticmethod
    def save_to_csv(data: pd.DataFrame, filename: str):
        """
        Сохраняет данные в CSV-файл.
        """
        if not data.empty:
            data.to_csv(filename, index=False, encoding="utf-8-sig")
            print(f"Данные сохранены в {filename}.")
        else:
            print("Нет данных для сохранения.")

    @staticmethod
    def analyze_data(data: pd.DataFrame, ticker: str):
        """
        Анализирует все данные, доступные в DataFrame.
        """
        if data.empty:
            print(f"Данные для {ticker} отсутствуют для анализа.")
            return

        print(f"\n### Общая информация о данных для {ticker} ###")
        print(data.info())

        print(f"\n### Описание числовых данных для {ticker} ###")
        print(data.describe())

        print(f"\n### Проверка на пропущенные значения для {ticker} ###")
        missing_values = data.isnull().sum()
        print(missing_values[missing_values > 0])

    @staticmethod
    def analyze_prices(data: pd.DataFrame, ticker: str):
        """
        Анализ цен акций: средняя, минимальная, максимальная цена.
        """
        print(f"\n### Анализ цен для {ticker} ###")
        if "CLOSE" in data.columns:
            print(data["CLOSE"].describe())

            # Процентное изменение цены (если возможно)
            if "PREVWAPRICE" in data.columns:
                data["PRICE_CHANGE_%"] = (
                    (data["CLOSE"] - data["PREVWAPRICE"]) / data["PREVWAPRICE"] * 100
                )
                print(f"\nПроцентное изменение цены для {ticker}:")
                print(data[["TRADEDATE", "PRICE_CHANGE_%"]])
            else:
                print(f"Колонка 'PREVWAPRICE' отсутствует, расчет процентного изменения для {ticker} невозможен.")
        else:
            print(f"Данные для {ticker} не содержат необходимого столбца 'CLOSE'.")

    @staticmethod
    def calculate_moving_averages(data: pd.DataFrame, ticker: str):
        """
        Рассчитывает скользящие средние и находит точки пересечения.
        """
        if "CLOSE" not in data.columns:
            print(f"Данные для {ticker} не содержат столбца 'CLOSE' для расчёта скользящих средних.")
            return

        data["SMA_10"] = data["CLOSE"].rolling(window=10).mean()
        data["SMA_50"] = data["CLOSE"].rolling(window=50).mean()

        # Выявление точек пересечения (золотые и мертвые кресты)
        data["CROSS"] = (data["SMA_10"] > data["SMA_50"]).astype(int).diff()
        golden_crosses = data[data["CROSS"] == 1]
        death_crosses = data[data["CROSS"] == -1]

        print(f"\n### Скользящие средние для {ticker} ###")
        print(f"Золотые кресты (дата пересечения): {golden_crosses['TRADEDATE'].tolist()}")
        print(f"Мертвые кресты (дата пересечения): {death_crosses['TRADEDATE'].tolist()}")

        # Построение графика
        plt.figure(figsize=(10, 5))
        plt.plot(data["TRADEDATE"], data["CLOSE"], label="Цена закрытия", color="blue")
        plt.plot(data["TRADEDATE"], data["SMA_10"], label="SMA 10", color="green")
        plt.plot(data["TRADEDATE"], data["SMA_50"], label="SMA 50", color="red")
        plt.title(f"Скользящие средние для {ticker}")
        plt.xlabel("Дата")
        plt.ylabel("Цена")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def calculate_volatility(data: pd.DataFrame, ticker: str):
        """
        Рассчитывает волатильность и строит график.
        """
        if "CLOSE" not in data.columns:
            print(f"Данные для {ticker} не содержат столбца 'CLOSE' для расчёта волатильности.")
            return

        data["VOLATILITY"] = data["CLOSE"].rolling(window=10).std()

        print(f"\n### Волатильность для {ticker} ###")
        print(data[["TRADEDATE", "VOLATILITY"]].dropna())

        # Построение графика волатильности
        plt.figure(figsize=(10, 5))
        plt.plot(data["TRADEDATE"], data["VOLATILITY"], label="Волатильность (10-дневная)", color="purple")
        plt.title(f"Волатильность для {ticker}")
        plt.xlabel("Дата")
        plt.ylabel("Волатильность")
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def calculate_daily_returns(data: pd.DataFrame, ticker: str):
        """
        Рассчитывает дневную доходность и строит её распределение.
        """
        if "CLOSE" not in data.columns:
            print(f"Данные для {ticker} не содержат столбца 'CLOSE' для расчёта доходности.")
            return

        data["DAILY_RETURN"] = data["CLOSE"].pct_change() * 100

        print(f"\n### Дневная доходность для {ticker} ###")
        print(data[["TRADEDATE", "DAILY_RETURN"]].dropna())

        # Построение гистограммы доходности
        plt.figure(figsize=(8, 5))
        plt.hist(data["DAILY_RETURN"].dropna(), bins=20, color="orange", edgecolor="black")
        plt.title(f"Распределение дневной доходности для {ticker}")
        plt.xlabel("Доходность (%)")
        plt.ylabel("Частота")
        plt.grid(True)
        plt.show()