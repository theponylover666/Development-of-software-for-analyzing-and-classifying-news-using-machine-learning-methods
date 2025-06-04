import streamlit as st
import requests
import time
from datetime import datetime
from PIL import Image
from io import BytesIO
import pandas as pd

# === Константы ===
API_BASE = "http://localhost:8000"
COMPANIES = ["Сбербанк", "Газпром", "Лукойл", "ВТБ", "Роснефть", "Магнит", "МТС", "Аэрофлот"]

# Отображение разделов
section_map = {
    "russia": "🇷🇺 Россия",
    "world": "🌍 Мир",
    "business": "💼 Экономика",
    "culture": "🎭 Культура",
    "moscow": "🏙 Москва",
    "digital": "🖥 Digital",
    "photo": "📸 Фото",
    "sport": "🏅 Спорт"
}

# === Интерфейс Streamlit ===
st.set_page_config(layout="wide")
st.title("Анализ новостей и динамики акций")

# === Панель параметров ===
with st.sidebar:
    st.header("Параметры анализа")
    company = st.selectbox("Выберите компанию", COMPANIES)
    today = datetime.today().date()
    start_date = st.date_input("Дата начала", datetime(2025, 1, 1), max_value=today)
    end_date = st.date_input("Дата окончания", today, max_value=today)
    threshold = st.slider("Порог изменения цены (%)", 1.0, 10.0, 5.0, step=0.5)
    forecast_days = st.slider("Дней для прогноза", 3, 30, 10)

    mode = st.radio(
        "Режим работы",
        options=["analyze", "validate", "monthly"],
        format_func=lambda x: {
            "analyze": "Обычный анализ",
            "validate": "Проверка модели",
            "monthly": "Прогноз на месяц"
        }[x]
    )

    # Описание выбранного режима
    if mode == "validate":
        st.info("Последние 10% данных используются как тестовая выборка.")
    elif mode == "monthly":
        st.info("Прогноз на месяц: обучение на всех данных до последнего месяца.")
    else:
        st.info("Прогноз на заданное число дней от конца диапазона.")

    analyze_btn = st.button("Запустить анализ")

# === Запуск анализа ===
if analyze_btn:
    with st.spinner("Запуск анализа..."):
        payload = {
            "ticker_name": company,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "threshold": threshold,
            "forecast_days": forecast_days,
            "mode": mode
        }

        try:
            # Запуск фоновой задачи анализа
            res = requests.post(f"{API_BASE}/analyze/start", json=payload)
            if res.status_code != 200:
                st.error("Не удалось запустить анализ.")
                st.stop()

            task_id = res.json()["task_id"]
            status_box = st.empty()
            progress_bar = st.progress(0)
            log_box = st.empty()

            # Отслеживание прогресса
            for i in range(100):
                status_res = requests.get(f"{API_BASE}/analyze/status/{task_id}")
                if status_res.status_code != 200:
                    st.error("Ошибка получения статуса задачи.")
                    st.stop()

                data = status_res.json()
                status_box.markdown(f"**Статус:** {data['status']}")
                log_box.code("\n".join(data["logs"]), language="text")

                if data["status"] == "done":
                    progress_bar.progress(100)
                    break
                elif data["status"] == "error":
                    st.error("Ошибка во время анализа.")
                    st.stop()
                else:
                    progress = min(95, int((i / 100) * 95))
                    progress_bar.progress(progress)
                    time.sleep(2)

            # Получение результата
            result_res = requests.get(f"{API_BASE}/analyze/result/{task_id}")
            if result_res.status_code != 200:
                st.error("Ошибка получения результатов.")
                st.stop()

            result = result_res.json()
            st.success("Анализ завершён!")

            # === Отображение результатов ===

            st.subheader("Краткий анализ цен")
            st.markdown(result["price_summary"])

            def render_news_block(title, items):
                st.subheader(title)
                if items:
                    for item in items:
                        section = item.get("section", "неизвестно")
                        section_pretty = section_map.get(section.lower(), section.capitalize())
                        st.markdown(
                            f"**{item['date']}** | {item['title']} [{item['label']}]  \n"
                            f"**Раздел:** *{section_pretty}*  \n"
                            f"[Читать новость]({item['url']})"
                        )
                else:
                    st.info("Нет новостей для отображения.")

            render_news_block("Новости по значимым дням", result["news"])
            render_news_block("Новости за последние 7 дней", result["recent_news"])

            st.subheader("Финальный вывод")
            st.markdown(result["summary"])

            if result.get("metrics"):
                st.subheader("Сравнение моделей")
                df_metrics = pd.DataFrame(result["metrics"])
                df_metrics["mae"] = df_metrics["mae"].round(3)
                df_metrics["rmse"] = df_metrics["rmse"].round(3)
                st.dataframe(df_metrics.sort_values("rmse"))

            st.subheader("Графики прогноза")
            for path in result["graph_paths"]:
                with open(path, "rb") as f:
                    img = Image.open(BytesIO(f.read()))
                    st.image(img, use_container_width=True)

        except Exception as e:
            st.exception(f"Ошибка: {e}")
