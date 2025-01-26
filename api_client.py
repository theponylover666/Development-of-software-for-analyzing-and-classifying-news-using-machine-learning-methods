import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

class APIClient:
    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Получение исторических данных акций с API Московской биржи.

        :param symbol: Тикер акции (например, "SBER", "GAZP").
        :param start_date: Начальная дата в формате "YYYY-MM-DD".
        :param end_date: Конечная дата в формате "YYYY-MM-DD".
        :return: DataFrame с колонками 'TRADEDATE' и 'CLOSE'. Если данные отсутствуют, возвращается пустой DataFrame.
        """
        try:
            # Формируем URL и параметры запроса
            url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.json"
            params = {'from': start_date, 'till': end_date}

            # Выполняем запрос
            response = requests.get(url, params=params)
            print(f"Запрос к API: {response.url}")

            # Проверяем успешность ответа
            if response.status_code == 200:
                data = response.json()

                # Извлечение данных из ответа
                if 'history' in data and 'data' in data['history']:
                    columns = data['history']['columns']
                    records = data['history']['data']

                    # Преобразование данных в DataFrame
                    if records:
                        df = pd.DataFrame(records, columns=columns)
                        if 'TRADEDATE' in df.columns and 'CLOSE' in df.columns:
                            print(f"Данные для {ticker} успешно получены.")
                            return df[['TRADEDATE', 'CLOSE']]
                        else:
                            print(f"Данные для {ticker} имеют неправильный формат.")
                            return pd.DataFrame()
                    else:
                        print(f"Нет данных для тикера {ticker} за указанный период.")
                        return pd.DataFrame()
                else:
                    print("Ответ API не содержит необходимых данных.")
                    return pd.DataFrame()
            else:
                print(f"Ошибка API {response.status_code}: {response.text}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return pd.DataFrame()