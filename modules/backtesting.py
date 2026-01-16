"""
Module de backtesting et calcul des métriques de performance
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .math_operations import arithmetic_returns, sharpe_ratio


def generate_sma_signals(data: pd.DataFrame, short_period: int = 20, long_period: int = 50) -> pd.Series:
    """
    Génère des signaux de trading basés sur le croisement de deux SMA
    
    Stratégie:
        - Signal d'achat (position = 1): SMA_short > SMA_long
        - Signal de vente (position = 0): SMA_short ≤ SMA_long
    
    Args:
        data: DataFrame avec colonne 'Close'
        short_period: Période de la SMA courte
        long_period: Période de la SMA longue
    
    Returns:
        Série de positions (0 ou 1)
    """
    from .technical_indicators import SMA
    
    sma_short = SMA(data['Close'], short_period)
    sma_long = SMA(data['Close'], long_period)
    
    # Générer les signaux
    signals = np.where(sma_short > sma_long, 1, 0)
    
    return pd.Series(signals, index=data.index, name='Position')


def calculate_strategy_returns(signals: pd.Series, returns: pd.Series, 
                               transaction_fee: float = 0.001) -> pd.Series:
    """
    Calcule les rendements de la stratégie
    
    Formule mathématique:
        R_strat(t) = Position(t-1) × R_actif(t) - Frais
    
    Args:
        signals: Série de positions (0 ou 1)
        returns: Série de rendements de l'actif
        transaction_fee: Frais de transaction (par défaut 0.1%)
    
    Returns:
        Série de rendements de la stratégie
    """
    # Décaler les positions d'une période (on trade sur le signal du jour précédent)
    positions = signals.shift(1).fillna(0)
    
    # Calculer les rendements de la stratégie
    strategy_returns = positions * returns
    
    # Appliquer les frais de transaction lors des changements de position
    position_changes = positions.diff().abs()
    fees = position_changes * transaction_fee
    
    # Soustraire les frais
    strategy_returns = strategy_returns - fees
    
    return strategy_returns


def portfolio_evolution(returns: pd.Series, initial_capital: float = 1000) -> pd.Series:
    """
    Calcule l'évolution du capital
    
    Formule mathématique:
        C(t) = C(t-1) × (1 + R_strat(t))
        ou de manière équivalente:
        C(t) = C_0 × Π(1 + R_strat(i))
    
    Args:
        returns: Série de rendements de la stratégie
        initial_capital: Capital initial
    
    Returns:
        Série d'évolution du capital
    """
    capital = initial_capital * (1 + returns).cumprod()
    return capital


def maximum_drawdown(capital: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calcule le Maximum Drawdown (MDD)
    
    Formule mathématique:
        MDD = max_{0≤t≤T} [(C_t - max_{0≤s≤t} C_s) / max_{0≤s≤t} C_s]
    
    Args:
        capital: Série d'évolution du capital
    
    Returns:
        Tuple (mdd, date_peak, date_trough)
    
    Interprétation:
        Le MDD représente la perte maximale depuis le pic historique.
        Un MDD de -20% signifie que le capital a chuté de 20% depuis son plus haut.
    """
    # Calculer le pic historique glissant
    running_max = capital.expanding().max()
    
    # Calculer le drawdown à chaque instant
    drawdown = (capital - running_max) / running_max
    
    # Trouver le maximum drawdown
    mdd = drawdown.min()
    
    # Trouver les dates du pic et du creux
    mdd_date = drawdown.idxmin()
    peak_date = capital[:mdd_date].idxmax()
    
    return mdd, peak_date, mdd_date


def calculate_trades(signals: pd.Series) -> pd.DataFrame:
    """
    Identifie tous les trades (entrées et sorties)
    
    Args:
        signals: Série de positions
    
    Returns:
        DataFrame avec les trades
    """
    # Identifier les changements de position
    position_changes = signals.diff()
    
    # Entrées (0 -> 1)
    entries = position_changes[position_changes == 1].index
    
    # Sorties (1 -> 0)
    exits = position_changes[position_changes == -1].index
    
    # Créer un DataFrame des trades
    trades = []
    for i, entry in enumerate(entries):
        # Trouver la sortie correspondante
        exit_candidates = exits[exits > entry]
        if len(exit_candidates) > 0:
            exit_date = exit_candidates[0]
            trades.append({
                'Entry Date': entry,
                'Exit Date': exit_date,
                'Duration': (exit_date - entry).days
            })
    
    return pd.DataFrame(trades)


def win_rate(strategy_returns: pd.Series, trades_df: pd.DataFrame) -> float:
    """
    Calcule le taux de réussite des trades
    
    Args:
        strategy_returns: Rendements de la stratégie
        trades_df: DataFrame des trades
    
    Returns:
        Taux de réussite (entre 0 et 1)
    """
    if len(trades_df) == 0:
        return 0.0
    
    winning_trades = 0
    for _, trade in trades_df.iterrows():
        trade_return = strategy_returns[trade['Entry Date']:trade['Exit Date']].sum()
        if trade_return > 0:
            winning_trades += 1
    
    return winning_trades / len(trades_df)


def profit_factor(strategy_returns: pd.Series) -> float:
    """
    Calcule le Profit Factor
    
    Formule mathématique:
        Profit Factor = Total des gains / Total des pertes
    
    Args:
        strategy_returns: Rendements de la stratégie
    
    Returns:
        Profit Factor
    
    Interprétation:
        - PF > 1: Stratégie profitable
        - PF > 2: Très bonne stratégie
        - PF < 1: Stratégie perdante
    """
    gains = strategy_returns[strategy_returns > 0].sum()
    losses = abs(strategy_returns[strategy_returns < 0].sum())
    
    if losses == 0:
        return np.inf if gains > 0 else 0
    
    return gains / losses


def performance_metrics(strategy_returns: pd.Series, capital: pd.Series, 
                       initial_capital: float = 1000, 
                       periods_per_year: int = 252) -> Dict[str, any]:
    """
    Calcule toutes les métriques de performance
    
    Args:
        strategy_returns: Rendements de la stratégie
        capital: Évolution du capital
        initial_capital: Capital initial
        periods_per_year: Nombre de périodes par an
    
    Returns:
        Dictionnaire avec toutes les métriques
    """
    # Rendement total
    total_return = (capital.iloc[-1] - initial_capital) / initial_capital
    
    # Rendement annualisé
    n_years = len(strategy_returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Volatilité annualisée
    volatility = strategy_returns.std() * np.sqrt(periods_per_year)
    
    # Ratio de Sharpe
    sharpe = sharpe_ratio(strategy_returns, periods_per_year=periods_per_year)
    
    # Maximum Drawdown
    mdd, peak_date, trough_date = maximum_drawdown(capital)
    
    # Profit Factor
    pf = profit_factor(strategy_returns)
    
    # Calmar Ratio (rendement annualisé / |MDD|)
    calmar = abs(annualized_return / mdd) if mdd != 0 else 0
    
    return {
        "Rendement total": total_return,
        "Rendement annualisé": annualized_return,
        "Volatilité annualisée": volatility,
        "Ratio de Sharpe": sharpe,
        "Maximum Drawdown": mdd,
        "Date du pic": peak_date,
        "Date du creux": trough_date,
        "Profit Factor": pf,
        "Calmar Ratio": calmar,
        "Capital final": capital.iloc[-1],
        "Nombre de périodes": len(strategy_returns)
    }


def backtest_sma_crossover(data: pd.DataFrame, 
                          short_period: int = 20, 
                          long_period: int = 50,
                          initial_capital: float = 1000,
                          transaction_fee: float = 0.001) -> Dict[str, any]:
    """
    Backtest complet de la stratégie SMA crossover
    
    Args:
        data: DataFrame OHLC
        short_period: Période de la SMA courte
        long_period: Période de la SMA longue
        initial_capital: Capital initial
        transaction_fee: Frais de transaction
    
    Returns:
        Dictionnaire avec tous les résultats du backtest
    """
    # Calculer les rendements de l'actif
    returns = arithmetic_returns(data['Close'])
    
    # Générer les signaux
    signals = generate_sma_signals(data, short_period, long_period)
    
    # Calculer les rendements de la stratégie
    strategy_returns = calculate_strategy_returns(signals, returns, transaction_fee)
    
    # Calculer l'évolution du capital
    capital = portfolio_evolution(strategy_returns, initial_capital)
    
    # Calculer les trades
    trades_df = calculate_trades(signals)
    
    # Calculer le taux de réussite
    wr = win_rate(strategy_returns, trades_df)
    
    # Calculer les métriques de performance
    metrics = performance_metrics(strategy_returns, capital, initial_capital)
    
    # Ajouter des métriques supplémentaires
    metrics["Nombre de trades"] = len(trades_df)
    metrics["Taux de réussite"] = wr
    
    # Créer un DataFrame avec tous les résultats
    results_df = data.copy()
    results_df['Returns'] = returns
    results_df['Position'] = signals
    results_df['Strategy_Returns'] = strategy_returns
    results_df['Capital'] = capital
    
    # Ajouter les SMA
    from .technical_indicators import SMA
    results_df[f'SMA_{short_period}'] = SMA(data['Close'], short_period)
    results_df[f'SMA_{long_period}'] = SMA(data['Close'], long_period)
    
    return {
        "data": results_df,
        "metrics": metrics,
        "trades": trades_df
    }


def compare_with_buy_and_hold(strategy_capital: pd.Series, 
                              asset_prices: pd.Series,
                              initial_capital: float = 1000) -> Dict[str, float]:
    """
    Compare la stratégie avec une stratégie Buy & Hold
    
    Args:
        strategy_capital: Évolution du capital de la stratégie
        asset_prices: Prix de l'actif
        initial_capital: Capital initial
    
    Returns:
        Dictionnaire avec la comparaison
    """
    # Calculer le rendement Buy & Hold
    bh_return = (asset_prices.iloc[-1] - asset_prices.iloc[0]) / asset_prices.iloc[0]
    bh_capital = initial_capital * (1 + bh_return)
    
    # Calculer le rendement de la stratégie
    strategy_return = (strategy_capital.iloc[-1] - initial_capital) / initial_capital
    
    # Calculer l'alpha (rendement excédentaire)
    alpha = strategy_return - bh_return
    
    return {
        "Buy & Hold Return": bh_return,
        "Buy & Hold Capital": bh_capital,
        "Strategy Return": strategy_return,
        "Strategy Capital": strategy_capital.iloc[-1],
        "Alpha": alpha,
        "Outperformance": "Oui" if alpha > 0 else "Non"
    }
