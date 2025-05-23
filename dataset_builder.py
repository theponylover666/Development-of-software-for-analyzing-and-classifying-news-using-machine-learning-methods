import pandas as pd
from api_client import APIClient
from text_preprocessor import TextPreprocessor
from tqdm import tqdm
from datetime import timedelta

def create_labeled_dataset(news_data: pd.DataFrame, stock_data: pd.DataFrame,
                           window_days: int = 3, threshold: float = 2.0) -> pd.DataFrame:
    stock_data = stock_data.copy()
    stock_data["TRADEDATE"] = pd.to_datetime(stock_data["TRADEDATE"])
    news_data["date"] = pd.to_datetime(news_data["date"])
    stock_data.set_index("TRADEDATE", inplace=True)
    stock_data.sort_index(inplace=True)

    stock_data["DAILY_RETURN"] = stock_data["CLOSE"].pct_change()
    stock_data["VOLATILITY"] = stock_data["DAILY_RETURN"].rolling(window=3).std()

    available_dates = set(stock_data.index.date)
    labeled = []
    skipped_dates = 0

    preprocessor = TextPreprocessor()

    for _, news in tqdm(news_data.iterrows(), total=news_data.shape[0], desc="Разметка новостей"):
        base_date = news["date"].date()
        title = news["title"]
        section = news.get("section", "unknown")

        # Проверяем в окне D-1, D, D+1
        found = False
        for offset in [-1, 0, 1]:
            news_date = base_date + timedelta(days=offset)
            if news_date not in available_dates:
                continue
            base_row = stock_data[stock_data.index.date == news_date]
            if base_row.empty:
                continue

            base_price = base_row["CLOSE"].iloc[0]
            future_data = stock_data[stock_data.index.date > news_date].iloc[:window_days]
            if future_data.empty or len(future_data) < 2:
                continue

            # Успешно нашли валидную дату
            found = True
            break

        if not found:
            skipped_dates += 1
            continue

        future_price = future_data["CLOSE"].mean()
        price_change = ((future_price - base_price) / base_price) * 100
        label = 1 if price_change > threshold else (-1 if price_change < -threshold else 0)

        vol_start = base_row["VOLATILITY"].iloc[0]
        vol_future = future_data["VOLATILITY"].mean()
        vol_change = (vol_future - vol_start) if pd.notna(vol_start) else 0
        label_volatility = 1 if vol_change > 0.01 else 0

        trend_score = (future_data["CLOSE"].diff() > 0).mean()
        trend_type = "uptrend" if trend_score > 0.6 else "downtrend" if trend_score < 0.4 else "none"

        impact_type = []
        if trend_type in {"uptrend", "downtrend"}:
            impact_type.append("trend")
        if label_volatility:
            impact_type.append("volatility")
        impact_type_str = ",".join(impact_type) if impact_type else "none"

        if label == 1:
            impact_class = "up_vol" if label_volatility else "up"
        elif label == -1:
            impact_class = "down_vol" if label_volatility else "down"
        else:
            impact_class = "neutral_vol" if label_volatility else "neutral"

        # Расчёт признаков
        sentiment = preprocessor.analyze_sentiment(title)
        processed_title = preprocessor.preprocess(title)
        title_length = len(title)
        num_words = len(processed_title.split())

        labeled.append({
            "date": news_date,
            "title": title,
            "label": label,
            "label_volatility": label_volatility,
            "impact_type": impact_type_str,
            "impact_class": impact_class,
            "section": section,
            "sentiment": sentiment,
            "title_len": title_length,
            "num_words": num_words,
            "clean_title": processed_title
        })

    df = pd.DataFrame(labeled)
    print(f"\nРазмечено новостей: {len(df)}")
    print(f"Пропущено новостей: {skipped_dates}")
    print("\nРаспределение по классам:")
    print(df["impact_class"].value_counts())

    return df

def build_and_merge_datasets(tickers, start_date, end_date, threshold=0.5):
    all_datasets = []
    processed_dates = set()

    for ticker in tickers:
        print(f"\n=== Обработка {ticker} ===")
        api = APIClient()

        stock_data = api.fetch_stock_data(ticker, start_date, end_date)
        if stock_data.empty:
            print(f"Нет данных по акциям для {ticker}")
            continue

        significant_days = api.detect_significant_changes(stock_data, threshold)
        if significant_days.empty:
            print(f"Нет значительных изменений для {ticker}")
            continue

        news_data = api.fetch_news_for_significant_days(ticker, stock_data, threshold)
        news_data["date"] = pd.to_datetime(news_data["date"].str[:10])
        news_data = news_data[~news_data["date"].isin(processed_dates)]
        processed_dates.update(news_data["date"].tolist())

        if news_data.empty:
            print(f"Нет новых новостей для {ticker}")
            continue

        labeled = create_labeled_dataset(news_data, stock_data)
        if labeled.empty:
            print(f"Не удалось разметить новости для {ticker}")
            continue

        labeled["ticker"] = ticker
        all_datasets.append(labeled)
        labeled.to_csv(f"data/labeled_news_{ticker}_{start_date}_to_{end_date}.csv", index=False, encoding="utf-8-sig")

    if all_datasets:
        combined = pd.concat(all_datasets, ignore_index=True)
        combined.to_csv("data/labeled_news_ALL.csv", index=False, encoding="utf-8-sig")
        print("\nОбъединённый датасет сохранён: data/labeled_news_ALL.csv")
    else:
        print("Не удалось собрать ни одного датасета.")

def main():
    tickers = [
        "SBER", "GAZP", "LKOH", "YNDX", "ROSN", "TATN", "VTBR", "MGNT", "NVTK",
        "GMKN", "CHMF", "ALRS", "POLY", "AFKS", "MOEX", "MTSS", "PHOR", "PLZL",
        "RUAL", "TRNFP", "AKRN", "AFLT", "IRAO"
    ]
    start_date = "2021-01-01"
    end_date = "2025-04-01"
    threshold = 2.0

    build_and_merge_datasets(tickers, start_date, end_date, threshold)

if __name__ == "__main__":
    main()