from app import MainApp

if __name__ == "__main__":
    app = MainApp()

    # Параметры, "GAZP", "LKOH"
    tickers = ["SBER"]
    start_date = "2024-01-01"
    end_date = "2025-01-01"

    for ticker in tickers:
        output_file = f"data/{ticker}_stock_data.csv"
        app.run(ticker, start_date, end_date, output_file)
