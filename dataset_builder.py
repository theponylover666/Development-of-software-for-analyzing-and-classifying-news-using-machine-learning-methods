from datetime import datetime, timedelta
import pandas as pd
import os
from api_client import APIClient
from data_processor import DataProcessor


def create_labeled_dataset(news_data: pd.DataFrame, stock_data: pd.DataFrame,
                           window_days: int = 3, threshold: float = 1.0) -> pd.DataFrame:
    """Создание размеченного датасета на основе новостей и цен акций."""
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

    for _, news in news_data.iterrows():
        news_date = news["date"].date()
        title = news["title"]
        section = news.get("section", "unknown")

        if news_date not in available_dates:
            skipped_dates += 1
            continue

        base_row = stock_data[stock_data.index.date == news_date]
        if base_row.empty:
            skipped_dates += 1
            continue
        base_price = base_row["CLOSE"].iloc[0]

        # --- Берём ближайшие N рабочих дней
        future_data = stock_data[stock_data.index.date > news_date]
        future_data = future_data.iloc[:window_days]

        if future_data.empty or len(future_data) < 2:
            skipped_dates += 1
            continue

        # --- Метка направления (рост/падение)
        future_price = future_data["CLOSE"].mean()
        price_change = ((future_price - base_price) / base_price) * 100
        label = 1 if price_change > threshold else (-1 if price_change < -threshold else 0)

        # --- Метка волатильности
        vol_start = base_row["VOLATILITY"].iloc[0]
        vol_future = future_data["VOLATILITY"].mean()
        vol_change = (vol_future - vol_start) if pd.notna(vol_start) else 0
        label_volatility = 1 if vol_change > 0.01 else 0

        # --- Тип тренда
        trend_days = future_data["CLOSE"].diff().dropna()
        trend_type = "uptrend" if all(trend_days > 0) else "downtrend" if all(trend_days < 0) else "none"

        impact_type = []
        if trend_type in {"uptrend", "downtrend"}:
            impact_type.append("trend")
        if label_volatility:
            impact_type.append("volatility")

        impact_type_str = ",".join(impact_type) if impact_type else "none"

        # --- Финальный класс
        if label == 1:
            impact_class = "up_vol" if label_volatility else "up"
        elif label == -1:
            impact_class = "down_vol" if label_volatility else "down"
        else:
            impact_class = "neutral_vol" if label_volatility else "neutral"

        labeled.append({
            "date": news_date,
            "title": title,
            "label": label,
            "label_volatility": label_volatility,
            "impact_type": impact_type_str,
            "impact_class": impact_class,
            "section": section
        })

    df = pd.DataFrame(labeled)

    print(f"\nРазмечено новостей: {len(df)}")
    print(f"Пропущено новостей (нет данных по дате или окну): {skipped_dates}")
    print("\nРаспределение по классам:")
    print(df["impact_class"].value_counts())
    return df

def build_dataset(ticker: str, start_date: str, end_date: str, threshold: float = 5.0):
    """Создание размеченного датасета по одному тикеру."""
    api = APIClient()
    processor = DataProcessor()

    stock_data = api.fetch_stock_data(ticker, start_date, end_date)
    if stock_data.empty:
        print("Не удалось получить данные по акциям.")
        return

    print("Данные по акциям загружены.")
    significant_days = api.detect_significant_changes(stock_data, threshold)
    if significant_days.empty:
        print("Нет значительных изменений.")
        return

    print(f"Найдено {len(significant_days)} значительных дат.")
    news_data = api.fetch_news_for_significant_days(ticker, stock_data, threshold)
    if news_data.empty:
        print("Новости не найдены.")
        return

    print(f"Собрано {len(news_data)} новостей.")
    labeled = create_labeled_dataset(news_data, stock_data)
    if labeled.empty:
        print("Не удалось разметить новости.")
        return

    os.makedirs("data", exist_ok=True)
    filename = f"data/labeled_news_{ticker}_{start_date}_to_{end_date}.csv"
    labeled.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Размеченный датасет сохранён в {filename}")


def build_and_merge_datasets(tickers, start_date, end_date, threshold=5.0):
    """Создаёт и объединяет размеченные датасеты по всем тикерам."""
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
        print("\n✅ Объединённый датасет сохранён: data/labeled_news_ALL.csv")
    else:
        print("Не удалось собрать ни одного датасета.")


def main():
    tickers = [
        "SBER", "GAZP", "LKOH", "YNDX", "ROSN", "TATN", "VTBR", "MGNT", "NVTK",
        "GMKN", "CHMF", "ALRS", "POLY", "AFKS", "MOEX", "MTSS", "PHOR", "PLZL",
        "RUAL", "TRNFP", "AKRN", "AFLT", "IRAO"
    ]
    start_date = "2021-01-01"
    end_date = "2025-04-20"
    threshold = 1.5

    build_and_merge_datasets(tickers, start_date, end_date, threshold)


if __name__ == "__main__":
    main()
