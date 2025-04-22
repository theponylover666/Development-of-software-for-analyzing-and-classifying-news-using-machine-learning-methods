import streamlit as st
import requests
import time
from datetime import datetime
from PIL import Image
from io import BytesIO

API_BASE = "http://localhost:8000"
COMPANIES = ["Сбербанк", "Газпром", "Лукойл", "Яндекс", "Роснефть"]
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

st.set_page_config(layout="wide")
st.title("📊 Анализ новостей и динамики акций")

with st.sidebar:
    st.header("Параметры анализа")
    company = st.selectbox("Выберите компанию", COMPANIES)
    start_date = st.date_input("Дата начала", datetime(2025, 2, 1))
    end_date = st.date_input("Дата окончания", datetime(2025, 3, 30))
    threshold = st.slider("Порог изменения цены (%)", 1.0, 10.0, 5.0, step=0.5)
    forecast_days = st.slider("Дней для прогноза (в режиме анализа)", 3, 30, 10)

    mode = st.radio(
        "Режим работы",
        ["Обычный анализ", "Проверка модели (валидация)"]
    )
    mode_map = {
        "Обычный анализ": "analyze",
        "Проверка модели (валидация)": "validate"
    }

    analyze_btn = st.button("Запустить анализ")


if analyze_btn:
    with st.spinner("Инициализация анализа..."):

        payload = {
            "ticker_name": company,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "threshold": threshold,
            "forecast_days": forecast_days,
            "mode": mode_map[mode]
        }

        try:
            res = requests.post(f"{API_BASE}/analyze/start", json=payload)
            if res.status_code != 200:
                st.error("Не удалось запустить анализ.")
                st.stop()

            task_id = res.json()["task_id"]

            status_box = st.empty()
            progress_bar = st.progress(0)
            log_box = st.empty()

            for i in range(100):
                status_res = requests.get(f"{API_BASE}/analyze/status/{task_id}")
                if status_res.status_code != 200:
                    st.error("Ошибка при получении статуса задачи.")
                    st.stop()

                data = status_res.json()
                status = data["status"]
                logs = data["logs"]

                log_box.code("\n".join(logs), language="text")
                status_box.markdown(f"**Статус:** {status}")

                if status == "done":
                    progress_bar.progress(100)
                    break
                elif status == "error":
                    st.error("Произошла ошибка во время анализа.")
                    st.stop()
                else:
                    progress = min(95, int((i / 100) * 95))
                    progress_bar.progress(progress)
                    time.sleep(2)

            # Получаем финальный результат
            result_res = requests.get(f"{API_BASE}/analyze/result/{task_id}")
            if result_res.status_code != 200:
                st.error("Ошибка при получении результата анализа.")
                st.stop()

            result = result_res.json()
            st.success("Анализ завершён!")

            # --- Краткий анализ цен
            st.subheader("📈 Краткий анализ цен")
            st.markdown(result["price_summary"])

            # --- Новости по значимым дням
            st.subheader("📰 Новости по значимым дням")
            if result["news"]:
                for item in result["news"]:
                    section = item.get("section", "неизвестно")
                    section_pretty = section_map.get(section.lower(), section.capitalize())
                    st.markdown(
                        f"**{item['date']}** | {item['title']} [{item['label']}]  \n"
                        f"**Раздел:** *{section_pretty}*  \n"
                        f"[Читать новость]({item['url']})"
                    )
            else:
                st.info("Нет новостей по значимым дням.")

            # --- Новости за последние 7 дней
            st.subheader("🗓 Новости за последние 7 дней")
            if result["recent_news"]:
                for item in result["recent_news"]:
                    section = item.get("section", "неизвестно")
                    section_pretty = section_map.get(section.lower(), section.capitalize())
                    st.markdown(
                        f"**{item['date']}** | {item['title']} [{item['label']}]  \n"
                        f"**Раздел:** *{section_pretty}*  \n"
                        f"[Читать новость]({item['url']})"
                    )
            else:
                st.info("Нет новостей за последние 7 дней.")

            # --- Финальный вывод
            st.subheader("📌 Финальный вывод")
            st.markdown(result["summary"])

            # --- Графики
            st.subheader("📊 Графики прогноза")
            for path in result["graph_paths"]:
                with open(path, "rb") as f:
                    img = Image.open(BytesIO(f.read()))
                    st.image(img, use_container_width=True)

        except Exception as e:
            st.exception(f"Ошибка: {e}")