import yfinance as yf
import pandas as pd


def import_crypto_data(symbol: str, period: str = "1y", interval: str = "1d"):
    """
    Importe l'historique des prix d'une cryptomonnaie depuis Yahoo Finance.

    :param symbol: Symbole de la crypto (ex: 'BTC-USD', 'ETH-USD')
    :param period: Durée de l'historique ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    :param interval: Granularité ('1m', '2m', '5m', '15m', '30m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    :return: DataFrame contenant l'historique des prix
    """
    try:
        data = yf.download(symbol, period=period, interval=interval)
        data.drop(columns=['Adj Close'], inplace=True, errors='ignore')  # On enlève Adj Close car souvent redondant
        data.dropna(inplace=True)  # Nettoyage des valeurs manquantes
        return data
    except Exception as e:
        print(f"Erreur lors de l'importation des données : {e}")
        return None
