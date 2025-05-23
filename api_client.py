import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta

TICKER_TO_COMPANY = {
    "SBER": ["Сбербанк", "Сбер", "Сбера", "Сбере"],
    "GAZP": ["Газпром", "Газпрома"],
    "LKOH": ["Лукойл", "Лукоил", "Лукойла"],
    "YNDX": ["Яндекс", "Yandex"],
    "ROSN": ["Роснефть", "Роснефти"],
    "TATN": ["Татнефть", "Татнефти"],
    "VTBR": ["ВТБ"],
    "MGNT": ["Магнит", "Магнита"],
    "NVTK": ["Новатэк", "Новатэка"],
    "GMKN": ["Норильский никель", "Норникель", "ГМК"],
    "CHMF": ["Северсталь"],
    "ALRS": ["Алроса", "АЛРОСА"],
    "POLY": ["Полюс", "Полюса"],
    "AFKS": ["Система", "АФК Система"],
    "MOEX": ["Мосбиржа", "MOEX"],
    "MTSS": ["МТС"],
    "PHOR": ["Фосагро", "ФосАгро"],
    "PLZL": ["Полиметалл", "Полюс Золото"],
    "RUAL": ["Русал", "РУСАЛ"],
    "TRNFP": ["Транснефть", "Транснефти"],
    "AKRN": ["Акрон"],
    "AFLT": ["Аэрофлот", "Aeroflot"],
    "IRAO": ["Интер РАО", "ИнтерРАО"],
}

class APIClient:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}

    @staticmethod
    def parse_interfax_section(url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.encoding = "windows-1251"
            soup = BeautifulSoup(response.text, "html.parser")

            meta_tag = soup.find("meta", {"property": "article:section"})
            if meta_tag and meta_tag.get("content"):
                return meta_tag["content"]
            aside = soup.find("aside", class_="textML")
            if aside:
                a_tag = aside.find("a")
                if a_tag and a_tag.text.strip():
                    return a_tag.text.strip()
        except Exception as e:
            print(f"[parser] Ошибка при определении раздела: {e}")
        return "Неизвестно"

    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Получение данных о ценах акций с MOEX.
        """
        url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.json"
        params = {'from': start_date, 'till': end_date, 'start': 0}
        all_records = []

        try:
            while True:
                response = requests.get(url, params=params)
                if response.status_code != 200:
                    print(f"Ошибка API {response.status_code}: {response.text}")
                    break

                data = response.json()
                if 'history' in data and 'data' in data['history']:
                    records = data['history']['data']
                    columns = data['history']['columns']
                    if not records:
                        break
                    all_records.extend(records)
                    if len(records) < 100:
                        break
                    params['start'] += 100
                else:
                    print("Ответ API не содержит данных.")
                    break

            if not all_records:
                print(f"Нет данных по {ticker} за указанный период.")
                return pd.DataFrame()

            df = pd.DataFrame(all_records, columns=columns)
            df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
            df["CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce")
            df = df.dropna(subset=["TRADEDATE", "CLOSE"])
            print(f"Данные для {ticker} успешно получены. Всего записей: {len(df)}")
            return df[["TRADEDATE", "CLOSE"]]
        except Exception as e:
            print(f"Ошибка получения данных: {e}")
            return pd.DataFrame()

    def detect_significant_changes(self, df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
        """
        Находит дни с наибольшими изменениями цен акций.
        """
        df = df.copy()
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
        df = df.groupby("TRADEDATE").agg({"CLOSE": "mean"}).reset_index()
        df['DAILY_CHANGE'] = df['CLOSE'].pct_change() * 100
        df = df.dropna()
        significant = df[abs(df['DAILY_CHANGE']) > threshold]
        print(f"Найдено {len(significant)} значительных изменений.")
        return significant[["TRADEDATE", "DAILY_CHANGE"]]

    def fetch_news_from_interfax(self, ticker: str, target_date: str) -> pd.DataFrame:
        """
        Парсит все новости за target_date и два дня до него.
        """
        base_url = "https://www.interfax.ru/news/"
        date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        date_range = [(date_obj - timedelta(days=i)).strftime("%Y/%m/%d") for i in range(2, -1, -1)]
        keywords = TICKER_TO_COMPANY.get(ticker, [])
        all_news = []

        for date in date_range:
            url = f"{base_url}{date}"
            try:
                response = requests.get(url, headers=self.headers)
                if response.status_code != 200:
                    print(f"Ошибка загрузки {url}: {response.status_code}")
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                articles = soup.find_all("div", attrs={"data-id": True})

                for article in articles:
                    try:
                        time_tag = article.find("span")
                        title_tag = article.find("h3")
                        link_tag = article.find("a")
                        if not (time_tag and title_tag and link_tag):
                            continue

                        news_time = f"{date} {time_tag.text.strip()}"
                        title = title_tag.text.strip()
                        url = "https://www.interfax.ru" + link_tag["href"]
                        section = link_tag["href"].strip("/").split("/")[0]

                        if any(kw.lower() in title.lower() for kw in keywords):
                            all_news.append({
                                "date": news_time,
                                "title": title,
                                "url": url,
                                "section": section
                            })
                    except Exception as e:
                        print(f"Ошибка парсинга: {e}")
                        continue
            except Exception as e:
                print(f"Ошибка загрузки страницы {date}: {e}")
                continue

        df = pd.DataFrame(all_news)
        print(f"Найдено {len(df)} новостей по {ticker} за {date_range}")
        return df

    def fetch_news_for_significant_days(self, ticker: str, df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
        """
        Получает новости на основе дней с сильными изменениями.
        Загружает страницы новостей, чтобы достать точный раздел (section).
        """
        significant = self.detect_significant_changes(df, threshold)
        all_news = pd.DataFrame()

        for _, row in significant.iterrows():
            target_date = row["TRADEDATE"].strftime("%Y-%m-%d")
            print(f"\nПарсинг новостей для {ticker} на дату {target_date}")
            news = self.fetch_news_from_interfax(ticker, target_date)

            if not news.empty:
                all_news = pd.concat([all_news, news], ignore_index=True)

        print(f"\nИтог: Найдено {len(all_news)} новостей для {len(significant)} значимых дней.")
        return all_news

    def fetch_news_from_interfax_range(self, ticker: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """
        Получает все новости за диапазон без ±2 дня.
        """
        all_news = []
        keywords = TICKER_TO_COMPANY.get(ticker, [])

        for offset in range((end_date - start_date).days + 1):
            date_str = (start_date + timedelta(days=offset)).strftime("%Y/%m/%d")
            url = f"https://www.interfax.ru/business/news/{date_str}"
            try:
                response = requests.get(url, headers=self.headers)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all("div", attrs={"data-id": True})

                for article in articles:
                    try:
                        time_tag = article.find("span")
                        title_tag = article.find("h3")
                        link_tag = article.find("a")
                        if not (time_tag and title_tag and link_tag):
                            continue

                        title = title_tag.text.strip()
                        url = "https://www.interfax.ru" + link_tag["href"]
                        section = link_tag["href"].strip("/").split("/")[0]
                        news_time = f"{date_str.replace('/', '-')} {time_tag.text.strip()}"

                        if any(kw.lower() in title.lower() for kw in keywords):
                            all_news.append({
                                "date": news_time,
                                "title": title,
                                "url": url,
                                "section": section
                            })
                    except:
                        continue
            except:
                continue

        return pd.DataFrame(all_news)

if __name__ == "__main__":
    client = APIClient()
    ticker = "SBER"
    test_date = (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d")

    print(f"\nПроверка подключения к interfax.ru для {test_date} ({ticker})")
    news = client.fetch_news_from_interfax(ticker, test_date)

    if news.empty:
        print("Новости не найдены или не удалось подключиться.")
    else:
        print(f"Найдено {len(news)} новостей. Примеры:")
        for i, row in news.head(5).iterrows():
            print(f"\n{row['date']}")
            print(f"Заголовок: {row['title']}")
            print(f"Раздел: {row.get('section', 'неизвестно')}")
            print(f"Ссылка: {row['url']}")
