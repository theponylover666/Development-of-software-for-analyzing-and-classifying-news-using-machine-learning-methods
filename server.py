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
from ml_models import LinearRegressionModel, ARIMAModel, SARIMAModel, SVRModel, KNNModel
import matplotlib
matplotlib.use("Agg")


# === Настройки ===
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
    "Роснефть": "ROSN"
}

# === Модели данных ===
class AnalyzeRequest(BaseModel):
    ticker_name: str
    start_date: str
    end_date: str
    threshold: float
    forecast_days: int
    mode: str = "analyze"

class NewsItem(BaseModel):
    date: str
    title: str
    label: str
    url: str
    section: str = "неизвестно"

class AnalyzeResponse(BaseModel):
    logs: List[str]
    news: List[NewsItem]
    recent_news: List[NewsItem]
    summary: str
    price_summary: str
    graph_paths: List[str]

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
        if stock_data.empty:
            raise Exception("Нет данных по акциям.")

        log(f"Запуск анализа для {ticker}...")
        log(f"Данные успешно получены. Всего записей: {len(stock_data)}")

        # === Подготовка данных ===
        if req.mode == "validate":
            split_idx = int(len(stock_data) * 0.9)
            train_data, test_data = stock_data.iloc[:split_idx], stock_data.iloc[split_idx:]
            forecast_days = len(test_data)
            log(f"Режим: validate. Обучение на {len(train_data)} днях, тест — {len(test_data)} дней.")
        else:
            train_data, test_data = stock_data.copy(), None
            forecast_days = req.forecast_days

        desc = stock_data["CLOSE"].describe()
        price_summary = (
            f"**Краткий анализ цен для {ticker}:**\n\n"
            f"Сводка по цене закрытия:\n"
            f"- Дней: {int(desc['count'])}\n"
            f"- Средняя: {desc['mean']:.2f} | Медиана: {desc['50%']:.2f}\n"
            f"- Мин: {desc['min']:.2f} | Макс: {desc['max']:.2f}\n"
            f"- Стд. отклонение: {desc['std']:.2f}"
        )

        def save_plot(fig, name):
            path = f"{TEMP_DIR}/{name}_{uuid.uuid4().hex[:8]}.png"
            fig.savefig(path)
            plt.close(fig)
            return path

        # === Прогнозирование
        graph_paths = []

        for name, Model in [("lr", LinearRegressionModel), ("arima", ARIMAModel),
                            ("sarima", SARIMAModel), ("svr", SVRModel), ("knn", KNNModel)]:
            model = Model(train_data)
            ts = train_data.set_index("TRADEDATE")["CLOSE"]

            if name == "arima":
                forecast = model.forecast(order=(1, 1, 1), forecast_days=forecast_days, return_data=True)
                forecast_values = [point["value"] for point in forecast]
                forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1),
                                               periods=len(forecast_values), freq="B")
                fig = model.visualize(ts, forecast_values, forecast_index=forecast_index, title="ARIMA")
            elif name == "sarima":
                forecast = model.forecast(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                          forecast_days=forecast_days, return_data=True)
                forecast_values = [point["value"] for point in forecast]
                forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1),
                                               periods=len(forecast_values), freq="B")
                fig = model.visualize(ts, forecast_values, forecast_index=forecast_index, title="SARIMA")
            else:
                fig = model.forecast(forecast_days)

            if fig:
                graph_paths.append(save_plot(fig, f"forecast_{name}"))

        # === Новости по значимым дням
        significant = main_app.api_client.detect_significant_changes(stock_data, req.threshold)
        log(f"\nНайдено {len(significant)} значительных изменений.")
        news_df = main_app.api_client.fetch_news_for_significant_days(ticker, stock_data, req.threshold)
        log(f"Итог: Найдено {len(news_df)} новостей для {len(significant)} значимых дней.")

        for _, row in news_df.iterrows():
            vec = prepare_vector(row["title"], row.get("section", "unknown"))
            pred = main_app.multi_model.predict(vec)[0]
            label = {1: "Рост", -1: "Падение"}.get(pred, "Нейтрально")
            news_output.append(NewsItem(
                date=row["date"],
                title=row["title"],
                label=label,
                url=row["url"],
                section=row.get("section", "неизвестно")
            ))

        # === Новости за последние 7 дней
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
            vectors = hstack([
                tfidf,
                [[s] for s in sentiments],
                [[sc] for sc in section_codes]
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
        TASKS[task_id]["result"] = AnalyzeResponse(
            logs=logs,
            news=news_output,
            recent_news=recent_output,
            summary=summary,
            price_summary=price_summary,
            graph_paths=graph_paths
        )
        TASKS[task_id]["status"] = "done"

    except Exception as e:
        log(f"Ошибка: {e}")
        TASKS[task_id]["status"] = "error"



def prepare_vector(title: str, section: str):
    """Предобработка и формирование вектора признаков одной новости."""
    processed = main_app.text_preprocessor.preprocess(title)
    sentiment = main_app.text_preprocessor.analyze_sentiment(title)
    section_code = main_app.section_encoder.transform([section])[0] \
        if section in main_app.section_encoder.classes_ else 0

    return hstack([
        main_app.vectorizer.transform([processed]),
        [[sentiment]],
        [[section_code]]
    ])
