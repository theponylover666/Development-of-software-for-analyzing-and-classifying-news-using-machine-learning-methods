import pandas as pd

# Загружаем основной датасет
main_path = "data/labeled_news_ALL.csv"
df_main = pd.read_csv(main_path)

# Ручные примеры
samples = [
    # Падение
    {"title": "Сбербанк понёс миллиардные убытки по итогам квартала", "impact_class": "down"},
    {"title": "Fitch понизил рейтинг Сбербанка", "impact_class": "down"},
    {"title": "Клиенты Сбербанка пожаловались на сбои в онлайн-банке", "impact_class": "down"},
    {"title": "Сбербанк оказался под новыми санкциями", "impact_class": "down"},
    {"title": "Снижение доходов по вкладам клиентов Сбербанка на 30%", "impact_class": "down"},
    {"title": "Сбербанк потерял долю на рынке кредитования", "impact_class": "down"},
    {"title": "Утечка данных в Сбербанке затронула миллионы клиентов", "impact_class": "down"},
    {"title": "Резкое падение прибыли Сбербанка в первом квартале", "impact_class": "down"},
    {"title": "Сбербанк сокращает персонал в нескольких регионах", "impact_class": "down"},
    {"title": "Отток вкладов из Сбербанка достиг рекордного уровня", "impact_class": "down"},

    # Рост
    {"title": "Сбербанк показал рекордную прибыль за квартал", "impact_class": "up"},
    {"title": "S&P повысило рейтинг Сбербанка", "impact_class": "up"},
    {"title": "Сбербанк расширил сеть офисов в регионах", "impact_class": "up"},
    {"title": "Сбербанк получил международную премию за инновации", "impact_class": "up"},
    {"title": "Резкий рост числа вкладчиков Сбербанка", "impact_class": "up"},
    {"title": "Сбербанк заключил стратегическое партнёрство с Huawei", "impact_class": "up"},
    {"title": "Спрос на ипотеку от Сбербанка вырос на 25%", "impact_class": "up"},
    {"title": "Сбербанк запустил новую платформу цифровых активов", "impact_class": "up"},
    {"title": "Чистая прибыль Сбербанка выросла на 30%", "impact_class": "up"},
    {"title": "Аналитики отмечают устойчивый рост бизнеса Сбербанка", "impact_class": "up"},
]

# Создаём DataFrame
df_samples = pd.DataFrame(samples)
df_samples["title_len"] = df_samples["title"].apply(len)
df_samples["num_words"] = df_samples["title"].apply(lambda x: len(x.split()))
df_samples["section"] = "Экономика"
df_samples["ticker"] = "SBER"

# Объединяем с основным датасетом
df_all = pd.concat([df_main, df_samples], ignore_index=True)

# Сохраняем результат
df_all.to_csv(main_path, index=False)
print(f"✅ Добавлено {len(df_samples)} примеров (up/down). Обновлённый файл сохранён в {main_path}")
