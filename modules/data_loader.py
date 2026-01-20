"""
Module d'acquisition des données financières
Supporte Yahoo Finance, Binance et fichiers CSV
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False


def load_from_yahoo(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Télécharge les données OHLC depuis Yahoo Finance
    
    Args:
        ticker: Symbole de l'actif (ex: 'AAPL', 'BTC-USD')
        start: Date de début (format: 'YYYY-MM-DD')
        end: Date de fin (format: 'YYYY-MM-DD')
        interval: Intervalle de temps ('1d', '1h', '5m', etc.)
    
    Returns:
        DataFrame avec colonnes: Open, High, Low, Close, Volume
    """
    try:
        # Nettoyer le ticker (enlever les espaces)
        ticker = ticker.strip().upper()
        
        # Créer un objet Ticker pour plus de contrôle
        ticker_obj = yf.Ticker(ticker)
        
        # Essayer de télécharger les données avec plusieurs méthodes
        data = None
        error_messages = []
        
        # Méthode 1: Utiliser yf.download (méthode standard)
        try:
            data = yf.download(
                ticker, 
                start=start, 
                end=end, 
                interval=interval, 
                progress=False,
                auto_adjust=True
            )
        except Exception as e1:
            error_messages.append(f"Méthode download: {str(e1)}")
        
        # Méthode 2: Utiliser ticker.history si la première méthode échoue
        if data is None or data.empty:
            try:
                data = ticker_obj.history(
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True
                )
            except Exception as e2:
                error_messages.append(f"Méthode history: {str(e2)}")
        
        # Vérifier si des données ont été récupérées
        if data is None or data.empty:
            error_msg = f"Aucune donnée trouvée pour {ticker} entre {start} et {end}.\n\n"
            error_msg += "⚠️ CAUSES POSSIBLES:\n\n"
            error_msg += "1. 📅 PLAGE DE DATES INVALIDE:\n"
            error_msg += f"   - Vérifiez que {start} et {end} sont des dates passées\n"
            error_msg += f"   - Date actuelle: 2026-01-20\n"
            error_msg += f"   - Assurez-vous que la date de début est avant la date de fin\n\n"
            error_msg += "2. 🎯 TICKER INCORRECT:\n"
            error_msg += f"   - Vérifiez que '{ticker}' existe sur Yahoo Finance\n"
            error_msg += "   - Essayez: AAPL, MSFT, GOOGL, BTC-USD\n\n"
            error_msg += "3. 🌐 PROBLÈMES DE CONNEXION:\n"
            error_msg += "   - Yahoo Finance peut être temporairement indisponible\n\n"
            error_msg += "4. ⏱️ INTERVALLE NON DISPONIBLE:\n"
            error_msg += f"   - L'intervalle '{interval}' peut ne pas être disponible pour cette période\n"
            if error_messages:
                error_msg += f"\n📋 DÉTAILS TECHNIQUES:\n" + "\n".join(f"   - {msg}" for msg in error_messages)
            raise ValueError(error_msg)
        
        # Nettoyer les données (supprimer les lignes avec NaN)
        initial_len = len(data)
        data = data.dropna()
        
        if len(data) == 0:
            raise ValueError(f"Toutes les données pour {ticker} contiennent des valeurs manquantes")
        
        # S'assurer que les colonnes sont correctes
        # Yahoo Finance peut retourner des colonnes avec ou sans majuscules
        column_mapping = {}
        for col in data.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                column_mapping[col] = 'Open'
            elif 'high' in col_lower:
                column_mapping[col] = 'High'
            elif 'low' in col_lower:
                column_mapping[col] = 'Low'
            elif 'close' in col_lower:
                column_mapping[col] = 'Close'
            elif 'volume' in col_lower:
                column_mapping[col] = 'Volume'
        
        # Renommer les colonnes si nécessaire
        if column_mapping:
            data = data.rename(columns=column_mapping)
        
        # Vérifier que toutes les colonnes requises sont présentes
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Colonnes manquantes: {', '.join(missing_columns)}. Colonnes disponibles: {', '.join(data.columns)}")
        
        # Garder uniquement les colonnes OHLCV
        data = data[required_columns]
        
        # Convertir en float pour éviter les problèmes de type
        data = data.astype(float)
        
        return data
    
    except ValueError as ve:
        # Propager les ValueError telles quelles
        raise Exception(f"Erreur lors du téléchargement depuis Yahoo Finance: {str(ve)}")
    
    except Exception as e:
        # Pour toutes les autres erreurs
        raise Exception(f"Erreur inattendue lors du téléchargement depuis Yahoo Finance: {str(e)}")


def load_from_binance(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Télécharge les données OHLC depuis Binance (cryptomonnaies)
    
    Args:
        symbol: Symbole de la paire (ex: 'BTCUSDT')
        start: Date de début (format: 'YYYY-MM-DD')
        end: Date de fin (format: 'YYYY-MM-DD')
        interval: Intervalle de temps ('1d', '1h', '5m', etc.)
    
    Returns:
        DataFrame avec colonnes: Open, High, Low, Close, Volume
    """
    if not BINANCE_AVAILABLE:
        raise ImportError("python-binance n'est pas installé. Utilisez: pip install python-binance")
    
    try:
        # Créer un client Binance (pas besoin de clés API pour les données publiques)
        client = Client("", "")
        
        # Mapper les intervalles
        interval_map = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
            "1wk": Client.KLINE_INTERVAL_1WEEK,
            "1mo": Client.KLINE_INTERVAL_1MONTH
        }
        
        binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_1DAY)
        
        # Télécharger les données
        klines = client.get_historical_klines(symbol, binance_interval, start, end)
        
        # Convertir en DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convertir les types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Garder uniquement les colonnes OHLC
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        
        return df
    
    except Exception as e:
        raise Exception(f"Erreur lors du téléchargement depuis Binance: {str(e)}")


def load_from_csv(filepath: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV
    
    Args:
        filepath: Chemin vers le fichier CSV
    
    Returns:
        DataFrame avec colonnes: Open, High, Low, Close, Volume
    
    Note:
        Le CSV doit avoir une colonne 'Date' ou un index datetime
        et les colonnes: Open, High, Low, Close, Volume
    """
    try:
        # Essayer de charger avec différents formats
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        except:
            df = pd.read_csv(filepath, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
        
        # Vérifier les colonnes requises
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Colonne manquante dans le CSV: {col}")
        
        # Nettoyer les données
        df = df.dropna()
        
        return df[required_columns]
    
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du CSV: {str(e)}")


def validate_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Valide les données OHLC
    
    Args:
        df: DataFrame à valider
    
    Returns:
        Tuple (is_valid, message)
    """
    # Vérifier que le DataFrame n'est pas vide
    if df.empty:
        return False, "Le DataFrame est vide"
    
    # Vérifier les colonnes requises
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Colonnes manquantes: {', '.join(missing_columns)}"
    
    # Vérifier qu'il n'y a pas de valeurs négatives
    if (df[['Open', 'High', 'Low', 'Close', 'Volume']] < 0).any().any():
        return False, "Valeurs négatives détectées"
    
    # Vérifier la cohérence OHLC (High >= Low)
    if (df['High'] < df['Low']).any():
        return False, "Incohérence détectée: High < Low"
    
    # Vérifier que High est le maximum
    if ((df['High'] < df['Open']) | (df['High'] < df['Close'])).any():
        return False, "Incohérence détectée: High n'est pas le maximum"
    
    # Vérifier que Low est le minimum
    if ((df['Low'] > df['Open']) | (df['Low'] > df['Close'])).any():
        return False, "Incohérence détectée: Low n'est pas le minimum"
    
    return True, "Données valides"


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Retourne des informations sur les données
    
    Args:
        df: DataFrame OHLC
    
    Returns:
        Dictionnaire avec les informations
    """
    return {
        "nombre_periodes": len(df),
        "date_debut": df.index[0],
        "date_fin": df.index[-1],
        "prix_min": df['Low'].min(),
        "prix_max": df['High'].max(),
        "volume_total": df['Volume'].sum(),
        "valeurs_manquantes": df.isnull().sum().sum()
    }
