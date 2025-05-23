from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime, timedelta
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import uuid
import os
import shutil
import pandas as pd
from app import MainApp
from ml_models import LinearRegressionModel, ARIMAModel, SARIMAModel, SVRModel, KNNModel, XGBoostModel
import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error
matplotlib.use("Agg")

app = FastAPI()
main_app = MainApp()
TEMP_DIR = "temp_graphs"
os.makedirs(TEMP_DIR, exist_ok=True)
TASKS: Dict[str, Dict] = {}

COMPANY_TO_TICKER = {
    "Сбербанк": "SBER",
    "Газпром": "GAZP",
    "Лукойл": "LKOH",
    "Яндекс": "YNDX",
    "Роснефть": "ROSN",
    "ВТБ": "VTBR",
    "Магнит": "MGNT",
    "МТС": "MTSS",
    "Аэрофлот": "AFLT"
}

def plot_forecast(ts, forecast_values, forecast_index, title="", ground_truth=None):
    import matplotlib.dates as mdates

    plt.figure(figsize=(12, 6))
    ts_index = pd.to_datetime(ts.index)
    forecast_index = pd.to_datetime(forecast_index)

    # Основные линии графика
    plt.plot(ts_index, ts.values, label="Исторические данные", color="blue", alpha=0.6)
    plt.plot(forecast_index, forecast_values, label="Прогноз", color="red", linestyle="--", marker="x")

    if ground_truth is not None:
        ground_truth = ground_truth[:len(forecast_values)]
        plt.plot(forecast_index, ground_truth.values, label="Фактические данные",
                 color="green", linestyle=":", marker="o")

    # Вертикальная линия конца тренировочных данных
    if len(ts_index) > 0:
        plt.axvline(ts_index[-1], color="gray", linestyle="--", alpha=0.5)
    # Оформление
    plt.title(title, fontsize=14)
    plt.xlabel("Дата", fontsize=12)
    plt.ylabel("Цена закрытия", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.xticks(rotation=45, fontsize=10)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.tight_layout()

    return plt.gcf()

# === Модели данных ===
class AnalyzeRequest(BaseModel):
    ticker_name: str
    start_date: str
    end_date: str
    threshold: float
    forecast_days: int
    mode: str = "monthly"

class NewsItem(BaseModel):
    date: str
    title: str
    label: str
    url: str
    section: str = "неизвестно"

class MetricItem(BaseModel):
    model: str
    mae: float
    rmse: float

class AnalyzeResponse(BaseModel):
    logs: List[str]
    news: List[NewsItem]
    recent_news: List[NewsItem]
    summary: str
    price_summary: str
    graph_paths: List[str]
    metrics: List[MetricItem] = []

class AnalyzeStartResponse(BaseModel):
    task_id: str

class AnalyzeStatusResponse(BaseModel):
    status: str
    logs: List[str]

# === Эндпоинты ===
@app.post("/analyze/start", response_model=AnalyzeStartResponse)
def start_analysis(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    ticker = COMPANY_TO_TICKER.get(req.ticker_name)
    if not ticker:
        raise HTTPException(status_code=400, detail="Неподдерживаемое имя компании.")

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "pending", "logs": [], "result": None}
    background_tasks.add_task(run_analysis, task_id, ticker, req)
    return AnalyzeStartResponse(task_id=task_id)

@app.get("/analyze/status/{task_id}", response_model=AnalyzeStatusResponse)
def get_status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return AnalyzeStatusResponse(status=task["status"], logs=task["logs"])

@app.get("/analyze/result/{task_id}", response_model=AnalyzeResponse)
def get_result(task_id: str):
    task = TASKS.get(task_id)
    if not task or task["status"] != "done":
        raise HTTPException(status_code=404, detail="Результат ещё не готов")
    return task["result"]

# === Основная логика анализа ===
def run_analysis(task_id: str, ticker: str, req: AnalyzeRequest):
    logs, news_output, recent_output = [], [], []

    def log(msg):
        print(msg)
        logs.append(msg)
        TASKS[task_id]["logs"] = logs.copy()

    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        os.makedirs(TEMP_DIR, exist_ok=True)

        stock_data = main_app.api_client.fetch_stock_data(ticker, req.start_date, req.end_date)
        if stock_data is None or stock_data.empty or "TRADEDATE" not in stock_data.columns:
            raise Exception("Нет данных по акциям или отсутствует колонка 'TRADEDATE'.")
        stock_data["TRADEDATE"] = pd.to_datetime(stock_data["TRADEDATE"])
        stock_data = stock_data.drop_duplicates(subset="TRADEDATE", keep="last").reset_index(drop=True)
        if stock_data.empty:
            raise Exception("Нет данных по акциям.")

        log(f"Запуск анализа для {ticker}...")
        log(f"Данные успешно получены. Всего записей: {len(stock_data)}")
        mode_label = "валидация" if req.mode == "validate" else "прогноз по месяцу"
        if req.mode == "validate":
            split_idx = int(len(stock_data) * 0.9)
            train_data, test_data = stock_data.iloc[:split_idx], stock_data.iloc[split_idx:]
            forecast_days = len(test_data)
            test_index = pd.to_datetime(test_data["TRADEDATE"])
            day_offset = train_data.shape[0]
            log(f"Режим: валидация. Обучение на {len(train_data)} днях, тест — {len(test_data)} дней.")

        elif req.mode == "monthly":
            stock_data.set_index("TRADEDATE", inplace=True)
            stock_data = stock_data.asfreq("B")
            stock_data["CLOSE"] = stock_data["CLOSE"].ffill()

            last_month = stock_data.index.max().to_period("M")
            train_data = stock_data[stock_data.index.to_period("M") < last_month].copy().reset_index()
            test_data = stock_data[stock_data.index.to_period("M") == last_month].copy().reset_index()

            forecast_days = len(test_data)
            test_index = pd.to_datetime(test_data["TRADEDATE"])
            day_offset = train_data.shape[0]
            log(f"Режим: по месяцу. Обучение на {len(train_data)} точках, тест на {len(test_data)} дней ({last_month})")

        else:
            train_data, test_data = stock_data.copy(), None
            forecast_days = req.forecast_days
            test_index = None
            day_offset = 0

        desc = stock_data["CLOSE"].describe()
        price_summary = (
            f"**Краткий анализ цен для {ticker}:**\n\n"
            f"Сводка по цене закрытия:\n"
            f"- Дней: {int(desc['count'])}\n"
            f"- Средняя: {desc['mean']:.2f} | Медиана: {desc['50%']:.2f}\n"
            f"- Мин: {desc['min']:.2f} | Макс: {desc['max']:.2f}\n"
            f"- Стд. отклонение: {desc['std']:.2f}"
        )

        ts = pd.Series(train_data["CLOSE"].values, index=pd.to_datetime(train_data["TRADEDATE"]))

        def save_plot(fig, name):
            path = f"{TEMP_DIR}/{name}_{uuid.uuid4().hex[:8]}.png"
            fig.savefig(path)
            plt.close(fig)
            return path

        graph_paths = []
        metrics = []
        for name, Model in [
            ("lr", LinearRegressionModel), ("arima", ARIMAModel), ("sarima", SARIMAModel),
            ("svr", SVRModel), ("xgb", XGBoostModel), ("knn", KNNModel)
        ]:
            model = Model(train_data) if name in {"arima", "sarima"} else Model(train_data, day_offset=day_offset)

            if req.mode in {"validate", "monthly"}:
                test_ts = pd.Series(test_data["CLOSE"].values, index=pd.to_datetime(test_data["TRADEDATE"]))
                test_ts = test_ts[~test_ts.index.duplicated(keep='last')]
                if name == "arima":
                    forecast = model.forecast(order=(1, 1, 1), forecast_days=forecast_days, return_data=True)
                    forecast_values = [point["value"] for point in forecast]
                    forecast_index = test_index[:len(forecast_values)]
                elif name == "sarima":
                    forecast = model.forecast(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                              forecast_days=forecast_days, return_data=True)
                    forecast_values = [point["value"] for point in forecast]
                    forecast_index = test_index[:len(forecast_values)]
                else:
                    tune = name in {"svr", "xgb"}
                    if name in {"lr", "svr", "knn", "xgb"}:
                        if tune:
                            forecast_data = model.forecast(forecast_days=forecast_days, return_data=True, tune=True)
                        else:
                            forecast_data = model.forecast(forecast_days=forecast_days, return_data=True)
                        forecast_values = [point["value"] for point in forecast_data]
                        forecast_index = test_index[:len(forecast_values)]
                    else:
                        forecast = model.forecast(order=(1, 1, 1), forecast_days=forecast_days,
                                                  return_data=True) if name == "arima" else \
                            model.forecast(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), forecast_days=forecast_days,
                                           return_data=True)
                        forecast_values = [point["value"] for point in forecast]
                        forecast_index = test_index[:len(forecast_values)]

                actual_values = test_ts.values[:len(forecast_values)]
                forecast_values = forecast_values[:len(actual_values)]
                mae = mean_absolute_error(actual_values, forecast_values)
                rmse = mean_squared_error(actual_values, forecast_values) ** 0.5
                log(f"{name.upper()} → MAE: {mae:.2f} | RMSE: {rmse:.2f}")
                metrics.append((name.upper(), mae, rmse))

                fig = plot_forecast(
                    ts,
                    forecast_values,
                    forecast_index[:len(forecast_values)],
                    title = f"{name.upper()} ({mode_label})",
                    ground_truth=test_ts
                )
            else:
                if name == "arima":
                    try:
                        fig = model.forecast(order=(1, 1, 1), forecast_days=forecast_days)
                    except Exception as e:
                        log(f"ARIMA: ошибка прогноза — {e}")
                        fig = None
                elif name == "sarima":
                    try:
                        fig = model.forecast(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), forecast_days=forecast_days)
                    except Exception as e:
                        log(f"SARIMA: ошибка прогноза — {e}")
                        fig = None
                elif name in {"svr", "xgb"}:
                    fig = model.forecast(forecast_days=forecast_days, tune=True)
                else:
                    fig = model.forecast(forecast_days=forecast_days)

            if hasattr(fig, "savefig"):
                graph_paths.append(save_plot(fig, f"forecast_{name}"))

        stock_data = stock_data.reset_index()
        significant = main_app.api_client.detect_significant_changes(stock_data, req.threshold)
        log(f"\nНайдено {len(significant)} значительных изменений.")
        news_df = main_app.api_client.fetch_news_for_significant_days(ticker, stock_data, req.threshold)
        log(f"Итог: Найдено {len(news_df)} новостей для {len(significant)} значимых дней.")

        for _, row in news_df.iterrows():
            vec = prepare_vector(row["title"], row.get("section", "unknown"), ticker)
            pred = main_app.multi_model.predict(vec)[0]
            label = {1: "Рост", -1: "Падение"}.get(pred, "Нейтрально")
            news_output.append(NewsItem(
                date=row["date"],
                title=row["title"],
                label=label,
                url=row["url"],
                section=row.get("section", "неизвестно")
            ))

        log(f"\nАнализ новостного фона за последние 7 дней для {ticker}:")
        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=7)
        recent_df = main_app.api_client.fetch_news_from_interfax_range(ticker, start_date, end_date)

        if not recent_df.empty:
            processed = [main_app.text_preprocessor.preprocess(t) for t in recent_df["title"]]
            sentiments = [main_app.text_preprocessor.analyze_sentiment(t) for t in recent_df["title"]]
            sections = [r.get("section", "unknown") for _, r in recent_df.iterrows()]
            section_codes = [
                main_app.section_encoder.transform([s])[0] if s in main_app.section_encoder.classes_ else 0
                for s in sections
            ]
            tfidf = main_app.vectorizer.transform(processed)

            ticker_code = main_app.ticker_encoder.transform([ticker])[0] \
                if ticker in main_app.ticker_encoder.classes_ else 0
            ticker_codes = [[ticker_code] for _ in range(len(recent_df))]
            title_lengths = [len(t) for t in recent_df["title"]]
            num_words = [len(p.split()) for p in processed]
            vectors = hstack([
                tfidf,
                [[s] for s in sentiments],
                [[sc] for sc in section_codes],
                ticker_codes,
                [[tl] for tl in title_lengths],
                [[nw] for nw in num_words]
            ])

            if vectors.shape[1] != main_app.multi_model.n_features_in_:
                vectors = csr_matrix(vectors)
                vectors = vectors[:, :main_app.multi_model.n_features_in_]

            preds = main_app.multi_model.predict(vectors)
            count_up = sum(preds == 1)
            count_down = sum(preds == -1)

            for i, row in enumerate(recent_df.itertuples()):
                label = {1: "Рост", -1: "Падение"}.get(preds[i], "Нейтрально")
                recent_output.append(NewsItem(
                    date=row.date,
                    title=row.title,
                    label=label,
                    url=row.url,
                    section=getattr(row, "section", "неизвестно")
                ))

            log(f"\nНайдено {len(recent_output)} новостей за 7 дней.")
            summary = f"Итог: Рост: {count_up}, Падение: {count_down}\n"
            summary += (
                "Новостной фон положительный — возможен рост." if count_up > count_down else
                "Новостной фон отрицательный — возможна просадка." if count_down > count_up else
                "Новостной фон нейтральный."
            )
        else:
            summary = "Нет новостей за последние 7 дней."

        # === Финальный ответ
        metrics_payload = [{"model": name, "mae": mae, "rmse": rmse} for name, mae, rmse in metrics]

        TASKS[task_id]["result"] = AnalyzeResponse(
            logs=logs,
            news=news_output,
            recent_news=recent_output,
            summary=summary,
            price_summary=price_summary,
            graph_paths=graph_paths,
            metrics=metrics_payload
        )
        TASKS[task_id]["status"] = "done"

    except Exception as e:
        log(f"Ошибка: {e}")
        TASKS[task_id]["status"] = "error"



def prepare_vector(title: str, section: str, ticker: str):
    processed = main_app.text_preprocessor.preprocess(title)
    sentiment = main_app.text_preprocessor.analyze_sentiment(title)

    section_code = main_app.section_encoder.transform([section])[0] \
        if section in main_app.section_encoder.classes_ else 0
    ticker_code = main_app.ticker_encoder.transform([ticker])[0] \
        if ticker in main_app.ticker_encoder.classes_ else 0

    title_len = len(title)
    num_words = len(processed.split())

    tfidf_vector = main_app.vectorizer.transform([processed])
    sentiment_vector = [[sentiment]]
    len_vector = [[title_len, num_words]]
    section_vector = [[section_code]]
    ticker_vector = [[ticker_code]]

    vector = hstack([
        tfidf_vector,
        sentiment_vector,
        len_vector,
        section_vector,
        ticker_vector
    ])

    expected = main_app.multi_model.n_features_in_
    actual = vector.shape[1]
    print(f"[DEBUG] Вектор признаков: {actual}, ожидается: {expected}")

    if actual != expected:
        diff = expected - actual
        print(f"[WARNING] Несоответствие размерности! Разница: {diff}")
        if diff > 0:
            from scipy.sparse import csr_matrix
            vector = hstack([vector, csr_matrix((1, diff))])
        else:
            vector = vector[:, :expected]

    return vector