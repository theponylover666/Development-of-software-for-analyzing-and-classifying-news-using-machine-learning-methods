import requests
import pandas as pd
from typing import Optional

class APIClient:
    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Получение исторических данных акций с API Московской биржи.

        :param ticker: Тикер акции (например, "SBER", "GAZP").
        :param start_date: Начальная дата в формате "YYYY-MM-DD".
        :param end_date: Конечная дата в формате "YYYY-MM-DD".
        :return: DataFrame с колонками 'TRADEDATE' и 'CLOSE'. Если данные отсутствуют, возвращается пустой DataFrame.
        """
        try:
            # Формируем URL и параметры запроса
            url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.json"
            params = {'from': start_date, 'till': end_date, 'start': 0}
            all_records = []  # Для хранения всех записей

            while True:
                # Выполняем запрос
                response = requests.get(url, params=params)
                print(f"Запрос к API: {response.url}")

                # Проверяем успешность ответа
                if response.status_code == 200:
                    data = response.json()

                    # Извлечение данных из ответа
                    if 'history' in data and 'data' in data['history']:
                        records = data['history']['data']
                        columns = data['history']['columns']

                        # Если данные есть, добавляем их
                        if records:
                            all_records.extend(records)
                            if len(records) < 100:  # Если данных меньше 100, больше страниц нет
                                break
                            else:
                                params['start'] += 100  # Переход на следующую страницу
                        else:
                            break
                    else:
                        print("Ответ API не содержит необходимых данных.")
                        break
                else:
                    print(f"Ошибка API {response.status_code}: {response.text}")
                    break

            # Если данные найдены, преобразуем их в DataFrame
            if all_records:
                df = pd.DataFrame(all_records, columns=columns)
                if 'TRADEDATE' in df.columns and 'CLOSE' in df.columns:
                    # Преобразуем дату и цену в нужный формат
                    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'], errors='coerce')
                    df['CLOSE'] = pd.to_numeric(df['CLOSE'], errors='coerce')
                    df = df.dropna(subset=['TRADEDATE', 'CLOSE'])  # Удаляем некорректные записи
                    print(f"Данные для {ticker} успешно получены. Всего записей: {len(df)}")
                    return df[['TRADEDATE', 'CLOSE']]
                else:
                    print("Формат данных некорректен.")
                    return pd.DataFrame()
            else:
                print(f"Нет данных для тикера {ticker} за указанный период.")
                return pd.DataFrame()
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return pd.DataFrame()
