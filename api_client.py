import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta

# Словарь ключевых слов для распознавания упоминаний компаний по тикеру
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
        """
        Определяет тематический раздел статьи Interfax по URL.
        """
        try:
            response = requests.get(url, timeout=10)
            response.encoding = "windows-1251"
            soup = BeautifulSoup(response.text, "html.parser")

            meta = soup.find("meta", {"property": "article:section"})
            if meta and meta.get("content"):
                return meta["content"]

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
        Загружает данные о ценах акций с MOEX по тикеру.
        """
        url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.json"
        params = {'from': start_date, 'till': end_date, 'start': 0}
        all_records = []

        try:
            while True:
                resp = requests.get(url, params=params)
                if resp.status_code != 200:
                    print(f"Ошибка API {resp.status_code}: {resp.text}")
                    break

                data = resp.json()
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
                    break

            if not all_records:
                return pd.DataFrame()

            df = pd.DataFrame(all_records, columns=columns)
            df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
            df["CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce")
            return df.dropna(subset=["TRADEDATE", "CLOSE"])[["TRADEDATE", "CLOSE"]]
        except Exception as e:
            print(f"Ошибка получения данных: {e}")
            return pd.DataFrame()

    def detect_significant_changes(self, df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
        """
        Определяет дни с изменением цены выше заданного порога (в %).
        """
        df = df.copy()
        df = df.groupby("TRADEDATE").agg({"CLOSE": "mean"}).reset_index()
        df["DAILY_CHANGE"] = df["CLOSE"].pct_change() * 100
        return df.dropna()[abs(df["DAILY_CHANGE"]) > threshold][["TRADEDATE", "DAILY_CHANGE"]]

    def fetch_news_from_interfax(self, ticker: str, target_date: str) -> pd.DataFrame:
        """
        Загружает новости Interfax за target_date и два дня до него,
        фильтрует по ключевым словам, связанным с тикером.
        """
        base_url = "https://www.interfax.ru/news/"
        date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        date_range = [(date_obj - timedelta(days=i)).strftime("%Y/%m/%d") for i in range(2, -1, -1)]
        keywords = TICKER_TO_COMPANY.get(ticker, [])
        all_news = []

        for date in date_range:
            try:
                resp = requests.get(f"{base_url}{date}", headers=self.headers)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                articles = soup.find_all("div", attrs={"data-id": True})

                for article in articles:
                    try:
                        time_tag = article.find("span")
                        title_tag = article.find("h3")
                        link_tag = article.find("a")
                        if not (time_tag and title_tag and link_tag):
                            continue

                        title = title_tag.text.strip()
                        if any(kw.lower() in title.lower() for kw in keywords):
                            all_news.append({
                                "date": f"{date} {time_tag.text.strip()}",
                                "title": title,
                                "url": "https://www.interfax.ru" + link_tag["href"],
                                "section": link_tag["href"].strip("/").split("/")[0]
                            })
                    except:
                        continue
            except:
                continue

        return pd.DataFrame(all_news)

    def fetch_news_for_significant_days(self, ticker: str, df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
        """
        Загружает новости по дням с резкими изменениями цен.
        """
        significant = self.detect_significant_changes(df, threshold)
        all_news = []

        for date in significant["TRADEDATE"]:
            news = self.fetch_news_from_interfax(ticker, date.strftime("%Y-%m-%d"))
            if not news.empty:
                all_news.append(news)

        return pd.concat(all_news, ignore_index=True) if all_news else pd.DataFrame()

    def fetch_news_from_interfax_range(self, ticker: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """
        Загружает все новости Interfax по тикеру в указанном диапазоне дат.
        """
        keywords = TICKER_TO_COMPANY.get(ticker, [])
        all_news = []

        for offset in range((end_date - start_date).days + 1):
            date_str = (start_date + timedelta(days=offset)).strftime("%Y/%m/%d")
            try:
                resp = requests.get(f"https://www.interfax.ru/business/news/{date_str}", headers=self.headers)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, 'html.parser')
                articles = soup.find_all("div", attrs={"data-id": True})

                for article in articles:
                    try:
                        time_tag = article.find("span")
                        title_tag = article.find("h3")
                        link_tag = article.find("a")
                        if not (time_tag and title_tag and link_tag):
                            continue

                        title = title_tag.text.strip()
                        if any(kw.lower() in title.lower() for kw in keywords):
                            all_news.append({
                                "date": f"{date_str.replace('/', '-')} {time_tag.text.strip()}",
                                "title": title,
                                "url": "https://www.interfax.ru" + link_tag["href"],
                                "section": link_tag["href"].strip("/").split("/")[0]
                            })
                    except:
                        continue
            except:
                continue

        return pd.DataFrame(all_news)
