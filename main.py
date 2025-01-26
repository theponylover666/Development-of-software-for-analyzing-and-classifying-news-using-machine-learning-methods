from app import MainApp

if __name__ == "__main__":
    app = MainApp()

    # Параметры
    tickers = ["SBER", "GAZP", "LKOH"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    for ticker in tickers:
        output_file = f"data/{ticker}_stock_data.csv"
        app.run(ticker, start_date, end_date, output_file)
