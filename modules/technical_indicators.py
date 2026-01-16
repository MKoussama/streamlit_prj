"""
Module d'indicateurs techniques
SMA, EMA, RSI, Bandes de Bollinger, MACD, ATR
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


def SMA(prices: pd.Series, n: int) -> pd.Series:
    """
    Moyenne Mobile Simple (Simple Moving Average)
    
    Formule mathématique:
        SMA_n(t) = (1/n) × Σ_{i=0}^{n-1} P_{t-i}
    
    Args:
        prices: Série de prix
        n: Période de la moyenne mobile
    
    Returns:
        Série de SMA
    
    Utilisation:
        - Identifier la tendance (prix > SMA = tendance haussière)
        - Support/Résistance dynamique
        - Croisements pour signaux de trading
    """
    return prices.rolling(window=n).mean()


def EMA(prices: pd.Series, n: int) -> pd.Series:
    """
    Moyenne Mobile Exponentielle (Exponential Moving Average)
    
    Formule mathématique:
        EMA_n(t) = α × P_t + (1 - α) × EMA_n(t-1)
        où α = 2 / (n + 1) (facteur de lissage)
    
    Args:
        prices: Série de prix
        n: Période de la moyenne mobile
    
    Returns:
        Série de EMA
    
    Avantage sur SMA:
        Donne plus de poids aux prix récents, donc réagit plus rapidement
        aux changements de prix.
    """
    return prices.ewm(span=n, adjust=False).mean()


def RSI(prices: pd.Series, n: int = 14) -> pd.Series:
    """
    Relative Strength Index
    
    Formule mathématique:
        RSI = 100 - (100 / (1 + RS))
        où RS = Moyenne des gains sur n périodes / Moyenne des pertes sur n périodes
    
    Args:
        prices: Série de prix
        n: Période du RSI (par défaut 14)
    
    Returns:
        Série de RSI (valeurs entre 0 et 100)
    
    Interprétation:
        - RSI > 70: Sur-acheté (possible retournement baissier)
        - RSI < 30: Sur-vendu (possible retournement haussier)
        - RSI = 50: Équilibre entre acheteurs et vendeurs
    """
    # Calculer les variations de prix
    delta = prices.diff()
    
    # Séparer les gains et les pertes
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculer les moyennes mobiles exponentielles des gains et pertes
    avg_gains = gains.ewm(span=n, adjust=False).mean()
    avg_losses = losses.ewm(span=n, adjust=False).mean()
    
    # Calculer le RS (Relative Strength)
    rs = avg_gains / avg_losses
    
    # Calculer le RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def bollinger_bands(prices: pd.Series, n: int = 20, k: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bandes de Bollinger
    
    Formules mathématiques:
        Bande moyenne = SMA_n(t)
        Bande supérieure = SMA_n(t) + k × σ_n(t)
        Bande inférieure = SMA_n(t) - k × σ_n(t)
    
    Args:
        prices: Série de prix
        n: Période de la SMA (par défaut 20)
        k: Nombre d'écarts-types (par défaut 2)
    
    Returns:
        Tuple (bande_superieure, bande_moyenne, bande_inferieure)
    
    Interprétation:
        - Prix touche bande supérieure: Sur-acheté
        - Prix touche bande inférieure: Sur-vendu
        - Bandes étroites: Faible volatilité (compression)
        - Bandes larges: Forte volatilité (expansion)
        
    Statistique:
        Avec k=2, environ 95% des prix devraient se situer entre les bandes
        (si distribution normale).
    """
    # Bande moyenne (SMA)
    middle_band = SMA(prices, n)
    
    # Écart-type glissant
    std = prices.rolling(window=n).std()
    
    # Bandes supérieure et inférieure
    upper_band = middle_band + k * std
    lower_band = middle_band - k * std
    
    return upper_band, middle_band, lower_band


def MACD(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence
    
    Formules mathématiques:
        MACD = EMA_fast(t) - EMA_slow(t)
        Signal = EMA_signal(MACD)
        Histogramme = MACD - Signal
    
    Args:
        prices: Série de prix
        fast: Période de l'EMA rapide (par défaut 12)
        slow: Période de l'EMA lente (par défaut 26)
        signal: Période de l'EMA du signal (par défaut 9)
    
    Returns:
        Tuple (macd, signal, histogram)
    
    Interprétation:
        - MACD > Signal: Momentum haussier
        - MACD < Signal: Momentum baissier
        - Croisement MACD/Signal: Signal de trading
        - Histogramme > 0: Force haussière
        - Histogramme < 0: Force baissière
    """
    # Calculer les EMA
    ema_fast = EMA(prices, fast)
    ema_slow = EMA(prices, slow)
    
    # Calculer le MACD
    macd = ema_fast - ema_slow
    
    # Calculer la ligne de signal
    signal_line = EMA(macd, signal)
    
    # Calculer l'histogramme
    histogram = macd - signal_line
    
    return macd, signal_line, histogram


def ATR(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """
    Average True Range - Mesure de la volatilité
    
    Formule mathématique:
        TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
        ATR = Moyenne mobile du TR sur n périodes
    
    Args:
        high: Série des prix hauts
        low: Série des prix bas
        close: Série des prix de clôture
        n: Période de l'ATR (par défaut 14)
    
    Returns:
        Série de ATR
    
    Utilisation:
        - Mesurer la volatilité du marché
        - Définir des stop-loss dynamiques
        - Ajuster la taille des positions
    """
    # Calculer le True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculer l'ATR (moyenne mobile du TR)
    atr = tr.ewm(span=n, adjust=False).mean()
    
    return atr


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Oscillateur Stochastique
    
    Formule mathématique:
        %K = 100 × (Close - Low_n) / (High_n - Low_n)
        %D = SMA(%K, d_period)
    
    Args:
        high: Série des prix hauts
        low: Série des prix bas
        close: Série des prix de clôture
        k_period: Période pour %K (par défaut 14)
        d_period: Période pour %D (par défaut 3)
    
    Returns:
        Tuple (%K, %D)
    
    Interprétation:
        - %K > 80: Sur-acheté
        - %K < 20: Sur-vendu
        - Croisement %K/%D: Signal de trading
    """
    # Calculer les plus hauts et plus bas sur k_period
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    # Calculer %K
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    
    # Calculer %D (moyenne mobile de %K)
    d = k.rolling(window=d_period).mean()
    
    return k, d


def get_all_indicators(data: pd.DataFrame, 
                       sma_periods: list = [20, 50],
                       ema_periods: list = [12, 26],
                       rsi_period: int = 14,
                       bollinger_period: int = 20,
                       macd_params: tuple = (12, 26, 9)) -> pd.DataFrame:
    """
    Calcule tous les indicateurs techniques sur un DataFrame OHLC
    
    Args:
        data: DataFrame avec colonnes Open, High, Low, Close, Volume
        sma_periods: Liste des périodes pour les SMA
        ema_periods: Liste des périodes pour les EMA
        rsi_period: Période pour le RSI
        bollinger_period: Période pour les Bandes de Bollinger
        macd_params: Tuple (fast, slow, signal) pour le MACD
    
    Returns:
        DataFrame avec tous les indicateurs ajoutés
    """
    df = data.copy()
    
    # SMA
    for period in sma_periods:
        df[f'SMA_{period}'] = SMA(df['Close'], period)
    
    # EMA
    for period in ema_periods:
        df[f'EMA_{period}'] = EMA(df['Close'], period)
    
    # RSI
    df['RSI'] = RSI(df['Close'], rsi_period)
    
    # Bandes de Bollinger
    upper, middle, lower = bollinger_bands(df['Close'], bollinger_period)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower
    
    # MACD
    macd, signal, histogram = MACD(df['Close'], *macd_params)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Histogram'] = histogram
    
    # ATR
    df['ATR'] = ATR(df['High'], df['Low'], df['Close'])
    
    return df
