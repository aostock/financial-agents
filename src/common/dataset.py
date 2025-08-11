import requests
import time
from urllib.parse import urlencode
from common.settings import Settings
from langchain_core.runnables import RunnableConfig


class Dataset:
    def __init__(self, config: RunnableConfig):
        self.settings = Settings(config)
        self.remote_dataset_url = self.settings.get_remote_financial_data_api_url().rstrip("/")
        self.remote_dataset_token = self.settings.get_remote_financial_data_api_key()
    
    def get_financial_metrics(self, symbol, end_date=None, period='quarterly'):
        data = self._request(f'ticker/financial_metrics', query={'symbol': symbol, 'freq': period})
        # filter end_date <= data['date']
        if end_date:
            data = [item for item in data if item['date'] <= end_date]
        return data
    
    def get_financial_items(self, symbol, items: list[str], end_date=None, period='quarterly'):
        if items is not None and len(items) > 0:
            items = ','.join(items)
        else:
            items = None
        data = self._request(f'ticker/financial_items', query={'symbol': symbol, 'items':items, 'freq': period})
        if end_date:
            data = [item for item in data if item['date'] <= end_date]
        return data
    
    def get_prices(self, symbol: str, start_date: str, end_date: str) -> list[dict]:
        return self._request(f'ticker/prices', query={'symbol': symbol, 'interval':'1d', 'start_date': start_date, 'end_date': end_date})
    
    def get_insider_transactions(self, symbol: str, end_date: str = None) -> list[dict]:
        data = self._request(f'ticker/insider_transactions', query={'symbol': symbol})
        # filter start_date <= end_date
        if end_date:
            data = [item for item in data if item['start_date'] <= end_date]
        return data
    
    def get_insider_roster_holders(self, symbol: str, end_date: str = None) -> list[dict]:
        data = self._request(f'ticker/insider_roster_holders', query={'symbol': symbol})
        # filter start_date <= end_date
        if end_date:
            data = [item for item in data if item['latest_transaction_date'] <= end_date]
        return data
    
    def get_news(self, symbol: str, end_date: str = None) -> list[dict]:
        data = self._request(f'ticker/news', query={'symbol': symbol, 'count': 200})
        # filter end_date <= data['pub_date']
        if end_date:
            data = [item for item in data if item['pub_date'] <= end_date]
        return data
    
    def get_info(self, symbol: str) -> dict:
        return self._request(f'ticker/info', query={'symbol': symbol})
    
    def lookup_ticker(self, query: str) -> dict:
        return self._request(f'ticker/lookup', query={'query': query})
    
    def _request(self, url: str, query: dict = None, max_retries: int = 3) -> requests.Response:
        # format url, trim leading '/'
        url = f'{self.remote_dataset_url}/api/v1/{url.lstrip("/")}'
        # format query string in url, and encode special characters
        if query:
            url += f'?{urlencode(query)}'
        
        headers = {
            'Authorization': f'Bearer {self.remote_dataset_token}'
        }
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            response = requests.get(url, headers=headers)
            if response.status_code != 200 and attempt < max_retries:
                delay = 10 + (5 * attempt)
                print(f"Request Failed. Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
                time.sleep(delay)
                continue
            result = response.json()
            if result['code'] != 0:
                raise Exception(f'Failed to get data from remote dataset. Url: {url}, Error: {result["code"]}, Msg: {result["msg"]}')
            return result['data']