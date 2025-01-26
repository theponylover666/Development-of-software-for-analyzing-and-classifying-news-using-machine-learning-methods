import pandas as pd
import matplotlib.pyplot as plt

class DataProcessor:
    """Класс для обработки и анализа данных, полученных с API."""

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
    def analyze_prices(data: pd.DataFrame):
        """
        Анализ цен акций: средняя, минимальная, максимальная цена и процентное изменение.
        """
        print("\n### Анализ цен ###")
        print(data[["PREVPRICE", "PREVWAPRICE"]].describe())

        # Процентное изменение цены
        data["PRICE_CHANGE_%"] = (
            (data["PREVPRICE"] - data["PREVWAPRICE"]) / data["PREVWAPRICE"] * 100
        )
        print("\nПроцентное изменение цены:")
        print(data[["SECID", "PRICE_CHANGE_%"]])

    @staticmethod
    def analyze_liquidity(data: pd.DataFrame):
        """
        Анализ ликвидности: LOTSIZE и ISSUESIZE.
        """
        print("\n### Анализ ликвидности ###")
        print("Размер лота (LOTSIZE):")
        print(data[["SECID", "LOTSIZE"]].describe())
        print("\nОбъём выпуска акций (ISSUESIZE):")
        print(data[["SECID", "ISSUESIZE"]].describe())

    @staticmethod
    def classify_parameters(data: pd.DataFrame):
        """
        Классификация компаний по параметрам: LISTLEVEL, CURRENCYID.
        """
        print("\n### Классификация по параметрам ###")
        print("Уровень листинга (LISTLEVEL):")
        print(data.groupby("LISTLEVEL")["SECID"].count())
        print("\nВалюта (CURRENCYID):")
        print(data.groupby("CURRENCYID")["SECID"].count())

    @staticmethod
    def compare_companies(data: pd.DataFrame):
        """
        Сравнение компаний по ценам и другим параметрам.
        """
        print("\n### Сравнение компаний ###")
        print("Сравнение цен (PREVPRICE):")
        print(data[["SECID", "PREVPRICE", "FACEVALUE", "LOTSIZE"]])

    @staticmethod
    def visualize_data(data: pd.DataFrame, ticker: str):
        """
        Визуализация данных.
        """
        if data.empty:
            print("Нет данных для визуализации.")
            return

        # График сравнения цен
        plt.figure(figsize=(10, 5))
        plt.plot(data["SECID"], data["PREVPRICE"], label="PrevPrice", marker="o")
        plt.plot(data["SECID"], data["PREVWAPRICE"], label="PrevWaPrice", marker="o")
        plt.title(f"Сравнение цен для {ticker}")
        plt.xlabel("SECID")
        plt.ylabel("Цена")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Гистограмма процентного изменения
        plt.figure(figsize=(8, 5))
        plt.bar(data["SECID"], data["PRICE_CHANGE_%"], color="skyblue")
        plt.title(f"Процентное изменение цен для {ticker}")
        plt.xlabel("SECID")
        plt.ylabel("Изменение (%)")
        plt.grid(True)
        plt.show()
