from binance.client import Client
import pandas as pd
import os
import time
import json
from datetime import datetime
from DataAnalysis.config import SYMBOLS, INTERVAL, START_DATE, DATA_PATH, USE_API_KEY, API_KEY, API_SECRET

META_PATH = os.path.join(DATA_PATH, "..", "meta.json")

def already_fetched_today():
    if not os.path.exists(META_PATH):
        return False
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    return meta.get("last_download_date") == datetime.utcnow().strftime('%Y-%m-%d')

def update_meta():
    with open(META_PATH, 'w') as f:
        json.dump({"last_download_date": datetime.utcnow().strftime('%Y-%m-%d')}, f)

def fetch_price_data():
    if already_fetched_today():
        print("✅ Données déjà à jour — téléchargement ignoré.")
        return

    client = Client(API_KEY, API_SECRET) if USE_API_KEY else Client()
    os.makedirs(DATA_PATH, exist_ok=True)

    for symbol in SYMBOLS:
        print(f"Downloading {symbol}")
        df = pd.DataFrame(client.get_historical_klines(symbol, INTERVAL, START_DATE))
        if not df.empty:
            df = df.iloc[:, :6]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.to_csv(os.path.join(DATA_PATH, f"{symbol}.csv"))
        time.sleep(1)

    update_meta()