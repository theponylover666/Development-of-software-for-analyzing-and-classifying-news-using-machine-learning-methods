from app import MainApp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    app = MainApp()
    tickers = ["SBER"]

    # Диапазон дат
    start_date = "2023-01-29"
    end_date = "2025-03-30"

    # Пороговое значение изменения цены (в %), при котором ищем новости
    threshold = 5.0

    # Количество дней для прогнозирования
    forecast_days = 10

    for ticker in tickers:
        output_file = f"data/{ticker}_stock_data.csv"
        app.run(ticker, start_date, end_date, output_file, threshold=threshold, forecast_days=forecast_days)

    # Показываем все графики после анализа
    plt.show()
